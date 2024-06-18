from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from keras.utils import to_categorical, plot_model
from module3.create_detaset import select_electrode, load_data, make_columns, generate_noise
from keras.models import load_model
from module2.create_model import one_dim_CNN_model, create_model_builder, line_notify, Bayesian_Opt, multi_stream_1D_CNN_model
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold, train_test_split
from skopt.space import Real, Categorical, Integer
import os
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import random
from collections import defaultdict
os.environ["OMP_NUM_THREADS"] = "1"


def generate_string(decompose_level, d_num):
    details = [f"d{i}" for i in range(decompose_level, decompose_level - d_num, -1)]
    # string is connected with "_"
    return "_".join(details)

def Preprocessing(preprocessing_type):
    if preprocessing_type == "d":
        preprocessing_dir = "DWT_data"
    elif preprocessing_type == "e":
        preprocessing_dir = "Envelope_data"
    elif preprocessing_type == "b" :
        preprocessing_dir = "BPF_data"
    return preprocessing_dir
# ////////////////////////////////////////////////////////////////////////////////////////
# left: 4662, right: 4608, fists: 4612, feet: 4643
n_class = 2 # How many class to classify (2 for left and right hands, 3 add for both hands, 4 add for both feet)
# number_of_chs = [64, 28] # How many channels to use (64, 38, 28, 19, 18, 12, 6)
number_of_ch = 64 # How many channels to use (64, 38, 28, 19, 18, 12, 6)
movement_types = ["left_right_fist", "fists_feet"] # static variables

extraction_section = True # True is not include rest section, False is include rest section
baseline_correction = True # baseline_correction(Basically True)
# create directory name
ext_sec = "move_only" if extraction_section else "rest_move"
baseline = "baseline_true" if baseline_correction else "baseline_false"

preprocessing_type = "d" # the kind of preprocessing method{d(DWT), e(Envelope), b(BPF)}
ds = 2 # down sampling rate(1/2, 1/3)

d_num = 3 # Number of detail to use(D4,D3...)(2 or 3)
decompose_level = 5 # Number of decomposition level

reduce_data = False # data reduction(True or False)
num_samples = 90  # Number of samples to use when reducing data(default=90)

sgkf = StratifiedGroupKFold(n_splits=5, random_state=22, shuffle=True) # cross validation for General model
sss_tl = StratifiedShuffleSplit(n_splits=4, random_state=22, test_size=0.2) # cross validation for Transfer model

ch_idx, extracted_ch = select_electrode(number_of_ch)
current_dir = Path.cwd()
preprocessing_dir = Preprocessing(preprocessing_type)

columns = make_columns(n_class)
subjects_dir = [current_dir/preprocessing_dir/ext_sec/baseline/f"S{i+1:03}" for i in range(104)]
subjects_name = [subject_dir.name for subject_dir in subjects_dir]

# select traget subject randomly
random.seed(22)
target_subjects = random.sample(subjects_name, 5)

details_dir = generate_string(decompose_level, d_num)

filename_change = "_multi_stream_4cla"

for target_subject in target_subjects:
    target_subject_index = target_subjects.index(target_subject)
    if n_class > 2:
        df = pd.DataFrame(index=[f"target_{target_subject}", "ACC", "PRE", "REC", "F1", "macro_REC", "macro_PRE", "macro_F1"], columns=columns)
    else:
        df = pd.DataFrame(index=[f"target_{target_subject}", "ACC", "PRE", "REC", "F1"], columns=columns)

    if preprocessing_dir == "DWT_data":
        data_paths = list((current_dir / "ML" / "ref_data" / preprocessing_dir / f"ds_{ds}" ).glob(f"S*/*.npy"))
    elif preprocessing_dir == "Envelope_data":
        data_paths = list((current_dir / "ML" / "ref_data" / preprocessing_dir / f"ds_{ds}" ).glob(f"S*/*.npy"))
    elif preprocessing_dir == "BPF_data":
        data_paths = list((current_dir / "ML" / "ref_data" / preprocessing_dir / f"ds_{ds}" ).glob(f"S*/*.npy"))

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

    # ベイズ最適化
    # best_params, _ = Bayesian_Opt(X, y, n_class, n_iter=150, cv=5)

    # Lists to store the accuracy of each fold
    train_acc_list = []
    val_acc_list = []

    # List to store confusion matrices
    conf_matrices = []

    # List to store recall, precision, F-value
    recalls, precisions, f_values = [], [], []
    conf_matrices_dict = {}

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
        model = one_dim_CNN_model(input_shape, n_class, optimizer='adam', learning_rate=0.001)
        # model = multi_stream_1D_CNN_model(input_shape, n_class, optimizer='adam', learning_rate=0.001)
        # plot_model(model, to_file="general_model.png", show_shapes=True)

        log_dir = current_dir / "ML" / "logs" / preprocessing_dir.split("_")[0] / f"{number_of_ch}ch" / f"ds_{ds}" / f"{n_class}class" / f"{fold+1}fold"
        model_dir = "ML" / Path("model_container")/preprocessing_dir.split('_')[0] / f"{number_of_ch}ch"/f"ds_{ds}"/f"{n_class}class" / f"{fold+1}fold"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        callbacks_list = [EarlyStopping(monitor="val_loss", patience=3),
                        ModelCheckpoint(f"{model_dir}/target_{target_subject}_model{filename_change}.keras", save_best_only=True, monitor='val_loss', mode='min'),
                        CSVLogger(log_dir / f"global_{target_subject}_model{filename_change}.csv", append=False)]
        # callbacks_list = [
        #                 ModelCheckpoint(f"{model_dir}/target_{target_subject}_model{filename_change}.keras", save_best_only=True, monitor='val_loss', mode='min'),
        #                 CSVLogger(log_dir / f"global_{target_subject}_model{filename_change}.csv", append=False)]

        model.fit(X_train_combined, y_train_combined, epochs=100, batch_size=12, validation_data=(X_test, y_test), callbacks=callbacks_list)

# //////////Transfer Learning part//////////
        group_results = []
        # load data of target subject
        X_sstl, y_sstl, _ = load_data([target_subject_data_path], movement_types, ch_idx, n_class, number_of_ch)
        # Convert labels to categorical
        y_sstl = to_categorical(y_sstl, num_classes=n_class)

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
        for train, test in sss_tl.split(X_sstl, y_sstl):
            X_train_sstl, X_test_sstl = X_sstl[train], X_sstl[test]
            y_train_sstl, y_test_sstl = y_sstl[train], y_sstl[test]

            # Validation data for SS-TL
            X_train_sstl, X_val, y_train_sstl, y_val = train_test_split(X_train_sstl, y_train_sstl, test_size=0.25, random_state=22, stratify=y_train_sstl)

            # Load General model
            general_model = load_model(f"{model_dir}/target_{target_subject}_model{filename_change}.keras")
            for layer in model.layers:
                if layer.name == "L4":
                    layer.trainable = True
                    break
                layer.trainable = False
            # # Output the trainable state of each layer
            # for layer in general_model.layers:
            #     print(layer.name, layer.trainable)
            # plot_model(general_model, to_file="SS-TL_model.png", show_shapes=True)

            sstl_callbacks_list = [EarlyStopping(monitor="val_loss", patience=3),
                                CSVLogger(log_dir / f"target_{target_subject}_model{filename_change}.csv", append=False)]
            # sstl_callbacks_list = [CSVLogger(log_dir / f"target_{target_subject}_model{filename_change}.csv", append=False)]

            # Transfer Learning
            history_sstl = general_model.fit(X_train_sstl, y_train_sstl, epochs=30, batch_size=12, validation_data=(X_val, y_val), callbacks=sstl_callbacks_list)
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
            conf_matrices.append(conf_matrix)
        conf_matrices_dict[f"{fold+1}fold"] = conf_matrices

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
        save_dir = current_dir / "ML" / "result"/ preprocessing_dir.split("_")[0] / "SS-TL_acc" / f"{number_of_ch}ch" / f"ds_{ds}" / f"{n_class}class"
        os.makedirs(save_dir, exist_ok=True)
    elif preprocessing_dir == "Envelope_data" or preprocessing_dir == "BPF_data":
        save_dir = current_dir / "ML" / "result"/ preprocessing_dir.split("_")[0] / "SS-TL_acc" / f"{number_of_ch}ch" / f"ds_{ds}" / f"{n_class}class"
        os.makedirs(save_dir, exist_ok=True)
    # Save the text file
    with open(save_dir / f"roc_auc_data_{target_subject}{filename_change}.txt", "w") as file:
        file.write(f"AUC: {auc}\n\n")  # AUC
        file.write("FPR\tTPR\tThresholds\n")
        for f, t, th in zip(fpr, tpr, thresholds):
            file.write(f"{f}\t{t}\t{th}\n")

    # Save the xlsx file
    save_path = save_dir / f"ave_evalute{filename_change}.xlsx"
    sheet_name = "SS-TL_model_evaluate"
    if target_subject_index == 0:
        df.to_excel(save_path, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(save_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            df.to_excel(writer, startrow=0, startcol=target_subject_index*(len(columns)+2), sheet_name=sheet_name)

    # Plot the heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(average_conf_matrix, annot=True, cmap="Blues", fmt=".1f",
                xticklabels=["Left Fist", "Right Fist"],
                yticklabels=["Left Fist", "Right Fist"])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Average Confusion Matrix Heatmap {target_subject}')
    plt.savefig(f"{save_dir}/{target_subject}_comf_martrix{filename_change}.png")
    plt.close()
    plt.clf()

message = "機械学習終了!!"
line_notify(message)
