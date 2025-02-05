from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.utils import to_categorical, plot_model
from module3.create_detaset import select_electrode, load_data, make_columns, generate_noise
from keras.models import load_model
from module2.create_model import one_dim_CNN_model, line_notify, multi_stream_1D_CNN_model
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold, train_test_split
import os
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import random
from collections import defaultdict
import tensorflow as tf
import time
from PIL import Image
from itertools import combinations

all_ch = ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6','Cp5', 'Cp3', 'Cp1',
        'Cpz', 'Cp2', 'Cp4', 'Cp6', 'Fp1', 'Fpz', 'Fp2', 'Af7', 'Af3', 'Afz', 'Af4','Af8', 'F7', 'F5', 'F3', 'F1','Fz',
        'F2', 'F4','F6', 'F8', 'Ft7', 'Ft8', 'T7', 'T8','T9', 'T10', 'Tp7', 'Tp8', 'P7', 'P5', 'P3', 'P1','Pz','P2',
        'P4','P6', 'P8', 'Po7','Po3', 'Poz', 'Po4', 'Po8', 'O1', 'Oz', 'O2', 'Iz']

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    # CPUスレッド数を制御
    os.environ['OMP_NUM_THREADS'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

def Preprocessing(preprocessing_type):
    if preprocessing_type == "d":
        preprocessing_dir = "DWT_data"
    elif preprocessing_type == "e":
        preprocessing_dir = "Envelope_data"
    elif preprocessing_type == "b" :
        preprocessing_dir = "BPF_data"
    return preprocessing_dir

def define_cD_index(extract_cD):
    cD_dict = {"D5": 1, "D4": 2, "D3": 3, "D2": 4, "D1": 5}
    cD_index = [cD_dict[cD] for cD in extract_cD]
    concatenated_string = "-".join(extract_cD)
    return cD_index, concatenated_string

def model_plot(model, model_image_filename):
    plot_model(model, to_file=model_image_filename, show_shapes=True, show_layer_names=True)
    im = Image.open(model_image_filename)
    plt.figure(figsize=(10, 8))
    plt.axis("off")
    plt.imshow(np.array(im))
    # plt.show()

def generate_artificial_eeg(data, labels, task_labels, num_trials, num_artificial):
    # initialize list to store artificial eeg
    artificial_eegs = []
    artificial_labels = []

    for task_label in task_labels:
        # get the indices of trials corresponding to the specified task label
        task_indices = [i for i, label in enumerate(labels) if label == task_label]

        # make all possible combinations of trials
        all_combinations = list(combinations(task_indices, num_trials))

        # check if the number of combinations is less than the number of artificial eeg to generate
        if num_artificial > len(all_combinations):
            raise ValueError(f"生成する人工脳波の数が多すぎます。タスク {task_label} には {len(all_combinations)} 通りの試行の組み合わせしかありません。")

        # randomly select the specified number of combinations
        selected_combinations = random.sample(all_combinations, num_artificial)

        # get the number of channels and samples
        num_channels = data.shape[2]
        num_samples = data.shape[1]

        # initialize array to store artificial eeg
        artificial_eeg = np.zeros((num_artificial, num_samples, num_channels))

        # generate each artificial eeg
        for i, combination in enumerate(selected_combinations):
            # generate superimposed data for each channel
            for channel in range(num_channels):
                # initialize the data to be superimposed
                superimposed_data = np.zeros(num_samples)

                # combine the data of the selected trials
                for idx in combination:
                    superimposed_data += data[idx, :, channel]

                # standardize after superimposition
                mean = np.mean(superimposed_data)
                std = np.std(superimposed_data)
                if std != 0:
                    superimposed_data = (superimposed_data - mean) / std

                # store in artificial eeg
                artificial_eeg[i, :, channel] = superimposed_data

        # append the generated artificial eeg and its label to the list
        artificial_eegs.append(artificial_eeg)
        artificial_labels.extend([task_label] * num_artificial)

    # convert list to numpy array
    artificial_eegs = np.concatenate(artificial_eegs, axis=0)

    return artificial_eegs, np.array(artificial_labels)

# setting environment variables
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # for using GPU
set_seed(93)
# ////////////////////////////////////////////////////////////////////////////////////////
# left: 4662, right: 4608, fists: 4612, feet: 4643
task_type = 0 # actual_imagine = 0, actual = 1, imagine = 2
task_name_list = ["actual_imagine", "actual", "imagine"]

num_class = [2, 3, 4] # How many class to classify (2 for left and right hands, 3 add for both hands, 4 add for both feet)
number_of_ch = 64 # How many channels to use
movement_types = ["left_right_fist", "fists_feet"] # static variables

extraction_section = True # True is not include rest section, False is include rest section
baseline_correction = True # baseline_correction(Basically True)
# create directory name
ext_sec = "move_only" if extraction_section else "rest_move"
baseline = "baseline_true" if baseline_correction else "baseline_false"

preprocessing_type = "d" # the kind of preprocessing method{d(DWT), e(Envelope), b(BPF)}
num_ds = [3] # down sampling rate(1/2, 1/3)

extract_cD = ["D5", "D4", "D3"]
cD_index, concatenated_string = define_cD_index(extract_cD)

reduce_data = False # data reduction(True or False)
num_samples = 90  # Number of samples to use when reducing data(default=90)

# paramete for artificial eeg
execute_artificial_eeg = 1 # if generate artificial eeg, set 1
num_trials = 5  # number of trials to combine
num_artificials = [50, 100, 150, 200, 250, 300]  # numebr of artificial eeg to generate(by each task)

sgkf = StratifiedGroupKFold(n_splits=5, random_state=22, shuffle=True) # cross validation for General model
sss_tl = StratifiedShuffleSplit(n_splits=4, random_state=22, test_size=0.2) # cross validation for Transfer model

ch_idx, extracted_ch = select_electrode(number_of_ch)
current_dir = Path.cwd()
preprocessing_dir = Preprocessing(preprocessing_type)

for num_artificial in num_artificials:
    filename_change = [f"_multi_4_L4_{num_artificial}_trial{num_trials}" if execute_artificial_eeg == 1 else "_multi_4_L4"][0]
    for n_class in num_class:
        for ds in num_ds:
            columns = make_columns(n_class)
            subjects_dir = [current_dir/preprocessing_dir/ext_sec/baseline/f"S{i+1:03}" for i in range(104)]
            subjects_name = [subject_dir.name for subject_dir in subjects_dir]

            # select traget subject randomly
            random.seed(22)
            target_subjects = random.sample(subjects_name, 5)

            # setting environment variables
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # for using GPU
            set_seed(93)

            time_df = pd.DataFrame(index=[target_subjects], columns=["time(s)"])
            for target_subject in target_subjects:
                target_subject_index = target_subjects.index(target_subject)
                if n_class > 2:
                    df = pd.DataFrame(index=[f"target_{target_subject}", "ACC", "PRE", "REC", "F1", "macro_REC", "macro_PRE", "macro_F1"], columns=columns)
                else:
                    df = pd.DataFrame(index=[f"target_{target_subject}", "ACC", "PRE", "REC", "F1"], columns=columns)
                if preprocessing_dir == "DWT_data":
                    data_paths = list((current_dir / "ML" / "ref_data" / preprocessing_dir / task_name_list[task_type] / concatenated_string / f"ds_{ds}" ).glob(f"S*/*.npy"))
                elif preprocessing_dir == "Envelope_data":
                    data_paths = list((current_dir / "ML" / "ref_data" / preprocessing_dir / task_name_list[task_type] / f"ds_{ds}" ).glob(f"S*/*.npy"))
                elif preprocessing_dir == "BPF_data":
                    data_paths = list((current_dir / "ML" / "ref_data" / preprocessing_dir / task_name_list[task_type] / f"ds_{ds}" ).glob(f"S*/*.npy"))

                for data_path in data_paths:
                    if data_path.parent.name == target_subject:
                        # data path of target subject
                        target_subject_data_path = data_path
                        # remove target subject data path from data_paths
                        data_paths.remove(data_path)
                        break

                X, y, subject_ids = load_data(data_paths, movement_types, ch_idx, n_class, number_of_ch)
                left_fist_idx = np.where(y == 0)
                right_fist_idx = np.where(y == 1)
                both_fists = np.where(y == 2)
                both_feet = np.where(y == 3)

                # Lists to store the accuracy of each fold
                train_acc_list = []
                val_acc_list = []

                # List to store confusion matrices
                conf_matrices = []

                # List to store recall, precision, F-value
                recalls, precisions, f_values = [], [], []
                conf_matrices_dict = {}
                execute_times = []
                # Split the dataset into training and testing sets for each fold
                for fold, (train_idxs, test_idxs) in enumerate(sgkf.split(X, y, groups=subject_ids)):
                    X_train, X_test = X[train_idxs], X[test_idxs]
                    y_train, y_test = y[train_idxs], y[test_idxs]

                    # generate noise and add to data
                    noise = generate_noise(X_train, "gauss", seed=22)
                    X_train_noised = X_train + noise

                    # Convert labels to categorical
                    y_train = to_categorical(y_train, num_classes=n_class)
                    y_test = to_categorical(y_test, num_classes=n_class)

                    X_train_combined = np.concatenate([X_train, X_train_noised])
                    y_train_combined = np.concatenate([y_train, y_train])

                    batch_size = X_train_combined.shape[0]
                    timesteps = X_train_combined.shape[1]
                    input_dim = X_train_combined.shape[2]
                    input_shape = (X_train_combined.shape[1], X_train_combined.shape[2])

                    # difinition of model
                    model = multi_stream_1D_CNN_model(input_shape, n_class, optimizer='adam', learning_rate=0.001)
                    # model_plot(model, "model.png")

                    log_dir = current_dir / "ML" / "logs" / preprocessing_dir.split("_")[0] / f"{number_of_ch}ch" / task_name_list[task_type] / concatenated_string / f"ds_{ds}" / f"{n_class}class" / f"{fold+1}fold"
                    model_dir = "ML" / Path("model_container")/preprocessing_dir.split('_')[0] / f"{number_of_ch}ch" / task_name_list[task_type] / concatenated_string / f"ds_{ds}"/f"{n_class}class" / f"{fold+1}fold"
                    os.makedirs(log_dir, exist_ok=True)
                    os.makedirs(model_dir, exist_ok=True)

                    callbacks_list = [EarlyStopping(monitor="val_loss", patience=3),
                                    ModelCheckpoint(f"{model_dir}/target_{target_subject}_model{filename_change}.keras", save_best_only=True, monitor='val_loss', mode='min'),
                                    CSVLogger(log_dir / f"global_{target_subject}_model{filename_change}.csv", append=False)]
                    start_time = time.time() # start time
                    model.fit(X_train_combined, y_train_combined, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks_list)

            # //////////Transfer Learning part//////////
                    group_results = []
                    # load data of target subject
                    X_sstl, y_sstl, _ = load_data([target_subject_data_path], movement_types, ch_idx, n_class, number_of_ch)

                    if execute_artificial_eeg == True:
                        # generate artificial eeg
                        combined_data = X_sstl.copy()
                        combined_labels = y_sstl.copy()

                        task_labels = list(range(n_class))
                        artificial_eeg, artificial_labels = generate_artificial_eeg(X_sstl, y_sstl, task_labels, num_trials, num_artificial)

                        combined_data = np.concatenate((combined_data, artificial_eeg), axis=0)
                        combined_labels = np.concatenate((combined_labels, artificial_labels))
                    else:
                        pass
                    # Convert labels to categorical
                    combined_labels = to_categorical(combined_labels, num_classes=n_class)

                    if reduce_data:
                        # generate random indices for reducing data
                        indices = np.arange(X_sstl.shape[0])
                        np.random.shuffle(indices, seed = 22)
                        indices = indices[:num_samples]

                        X_sstl = X_sstl[indices]
                        y_sstl = y_sstl[indices]
                    else:
                        pass

                    # Lists to store confusion matrices
                    conf_matrices = []
                    # List to store recall, precision, F-value
                    recalls, precisions, f_values = [], [], []
                    # Split the dataset into training and testing sets for each fold
                    for train, test in sss_tl.split(combined_data, combined_labels):
                        X_train_sstl, X_test_sstl = combined_data[train], combined_data[test]
                        y_train_sstl, y_test_sstl = combined_labels[train], combined_labels[test]

                        # Validation data for SS-TL
                        X_train_sstl, X_val, y_train_sstl, y_val = train_test_split(X_train_sstl, y_train_sstl, test_size=0.25, random_state=22, stratify=y_train_sstl)

                        # Load General model
                        general_model = load_model(f"{model_dir}/target_{target_subject}_model{filename_change}.keras")
                        for layer in general_model.layers:
                            if layer.name == "L4_1" or layer.name == "L4_2" or layer.name == "L4_3" or layer.name == "L4_4":
                                layer.trainable = True
                                break
                            layer.trainable = False
                        # Output the trainable state of each layer
                        # for layer in general_model.layers:
                        #     print(layer.name, layer.trainable)
                        # plot_model(general_model, to_file="SS-TL_model.png", show_shapes=True)

                        sstl_callbacks_list = [EarlyStopping(monitor="val_loss", patience=4),
                                            CSVLogger(log_dir / f"target_{target_subject}_model{filename_change}.csv", append=False)]

                        # Transfer Learning
                        history_sstl = general_model.fit(X_train_sstl, y_train_sstl, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=sstl_callbacks_list)

                        end_time = time.time() # end time
                        execute_times.append(end_time - start_time)
                        y_pred = general_model.predict(X_test_sstl)

                        y_pred_classes = np.argmax(y_pred, axis=1)
                        y_true_classes = np.argmax(y_test_sstl, axis=1)

                        # ROC curve
                        y_true_prob = y_test_sstl[:, 1]
                        y_pred_prob = y_pred[:, 1]
                        fpr, tpr, thresholds = roc_curve(y_true_prob, y_pred_prob)
                        # AUC score
                        auc = roc_auc_score(y_true_prob, y_pred_prob)

                        # Calculate confusion matrix
                        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
                        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
                        conf_matrices.append(conf_matrix_normalized)
                    conf_matrices_dict[f"{fold+1}fold"] = conf_matrices

                excute_average_time = round(np.mean(execute_times), 2)
                time_df.loc[target_subject] = excute_average_time
                average_conf_matrices = {}
                # Calculate average confusion matrix(Transfer Learning part)
                for fold, matrices in conf_matrices_dict.items():
                    average_conf_matrices[fold] = np.mean(matrices, axis=0)

                # Calculate average confusion matrix(General model)
                average_conf_matrix = np.mean(list(average_conf_matrices.values()), axis=0)

                n_classes = average_conf_matrix.shape[0]
                # variable for macro average
                macro_recall = 0
                macro_precision = 0
                macro_f_value = 0
                for i in range(n_classes):
                    TP = average_conf_matrix[i, i]
                    FP = sum(average_conf_matrix[:, i]) - TP
                    FN = sum(average_conf_matrix[i, :]) - TP
                    TN = sum(sum(average_conf_matrix)) - TP - FP - FN

                    accuracy = sum(average_conf_matrix.diagonal()) / sum(sum(average_conf_matrix))
                    recall = TP / (TP + FN)
                    precision = TP / (TP + FP)
                    f_value = 2 * precision * recall / (precision + recall)

                    macro_recall += recall
                    macro_precision += precision
                    macro_f_value += f_value

                    group_results.append({f"task{i}_ACC": f"{accuracy * 100:.2f}", f"task{i}_PRE": f"{precision * 100:.2f}",
                                        f"task{i}_REC": f"{recall * 100:.2f}", f"task{i}_F1": f"{f_value * 100:.2f}"})
                # Calculate macro average when multi-class classification
                if n_classes > 2:
                    macro_recall = macro_recall / n_classes
                    macro_precision = macro_precision / n_classes
                    macro_f_value = macro_f_value / n_classes

                    group_results.append({f"macro_PRE": f"{macro_precision * 100:.2f}", f"macro_REC": f"{macro_recall * 100:.2f}",
                                        f"macro_F1": f"{macro_f_value * 100:.2f}"})

                # Convert group_results_list to dict
                result = defaultdict(float)
                for group_result in group_results:
                    for k, v in group_result.items():
                        result[k] += float(v)
                result = dict(result)

                # Caluculate average of evaluation index of each task
                movetype = ["left_fist", "right_fist", "both_fists", "both_feet"]
                df_value = defaultdict(float)
                subjects_num = len([target_subject_data_path])
                for key, value in result.items():
                    if "macro" in key:
                        pass
                    else:
                        key = f"{movetype[int(key.split('_')[0][-1])]}_{key.split('_')[1]}"
                    df_value[key] = float(f"{value / subjects_num:.2f}")
                df_value = dict(df_value)

                # Writes the average of the evaluation index of each task to DataFrame
                for column in df_value.keys():
                    if "macro" in column:
                        df.loc[column] = df_value[column]
                    else:
                        split_column = column.split('_')
                        new_column = f"{split_column[0]}_{split_column[1]}"
                        df.loc[split_column[2], new_column] = df_value[column]

                # Output average of the evaluation index of each task
                if preprocessing_dir == "DWT_data":
                    save_dir = current_dir / "ML" / "result"/ preprocessing_dir.split("_")[0] / "SS-TL_acc" / f"{number_of_ch}ch" / task_name_list[task_type] / concatenated_string / f"ds_{ds}" / f"{n_class}class"
                    os.makedirs(save_dir, exist_ok=True)
                elif preprocessing_dir == "Envelope_data" or preprocessing_dir == "BPF_data":
                    save_dir = current_dir / "ML" / "result"/ preprocessing_dir.split("_")[0] / "SS-TL_acc" / f"{number_of_ch}ch" / task_name_list[task_type] / f"ds_{ds}" / f"{n_class}class"
                    os.makedirs(save_dir, exist_ok=True)
                # Save the text file
                with open(save_dir / f"roc_auc_data_{target_subject}{filename_change}.txt", "w") as file:
                    file.write(f"AUC: {auc}\n\n")  # AUC
                    file.write("FPR\tTPR\tThresholds\n")
                    for f, t, th in zip(fpr, tpr, thresholds):
                        file.write(f"{f}\t{t}\t{th}\n")

                # Save the xlsx file
                save_path = save_dir / f"ave_evalute{filename_change}.xlsx"
                time_save_path = save_dir / f"execute_time{filename_change}.xlsx"
                sheet_name = "SS-TL_model_evaluate"
                if target_subject_index == 0:
                    df.to_excel(save_path, sheet_name=sheet_name)
                    time_df.to_excel(time_save_path, sheet_name="time")
                else:
                    with pd.ExcelWriter(save_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                        df.to_excel(writer, startrow=0, startcol=target_subject_index*(len(columns)+2), sheet_name=sheet_name)

                    with pd.ExcelWriter(time_save_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                        time_df.to_excel(writer, startrow=0, sheet_name="time")

                # Plot the heatmap of the confusion matrix
                plt.figure(figsize=(8, 6))
                labels = {
                    2:["Left Fist", "Right Fist"],
                    3:["Left Fist", "Right Fist", "Both Fists"],
                    4:["Left Fist", "Right Fist", "Both Fists", "Both Feet"]
                }
                sns.heatmap(average_conf_matrix, annot=True, cmap="Blues", fmt=".1f",
                            xticklabels=labels[n_class],
                            yticklabels=labels[n_class])
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.title(f'Average Confusion Matrix Heatmap {target_subject}')
                plt.savefig(f"{save_dir}/{target_subject}_comf_martrix{filename_change}.png")
                plt.close()
                plt.clf()

                pd.DataFrame

message = "機械学習終了!!"
line_notify(message)
