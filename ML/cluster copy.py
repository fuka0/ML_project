from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from keras.utils import to_categorical, plot_model
from module3.create_detaset import select_electrode, load_data, make_columns, generate_noise
from keras.models import load_model
from module2.create_model import one_dim_CNN_model, create_model_builder, line_notify, Bayesian_Opt
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold, train_test_split
from skopt.space import Real, Categorical, Integer
import os
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import random
from collections import defaultdict
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
number_of_ch = 64 # How many channels to use (64, 38, 28, 19, 18, 12, 6)
movement_types = ["left_right_fist", "fists_feet"] # static variables

extraction_section = True # True is not include rest section, False is include rest section
baseline_correction = True # baseline_correction(Basically True)
# create directory name
ext_sec = "move_only" if extraction_section else "rest_move"
baseline = "baseline_true" if baseline_correction else "baseline_false"

preprocessing_type = "d" # the kind of preprocessing method{d(DWT), e(Envelope), b(BPF)}
ds = 2 # down sampling rate(1/2, 1/3)

filename_change = "_test"

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

details_dir = generate_string(decompose_level, d_num)
for target_subject in subjects_name:
    l8_outputs = []
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

    # Lists to store the accuracy of each fold
    train_acc_list = []
    val_acc_list = []

    # List to store confusion matrices
    conf_matrices = []

    # List to store recall, precision, F-value
    recalls, precisions, f_values = [], [], []
    conf_matrices_dict = {}

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
        # plot_model(model, to_file="general_model.png", show_shapes=True)

        log_dir = current_dir / "ML" / "logs" / preprocessing_dir.split("_")[0] / f"{number_of_ch}ch" / f"ds_{ds}" / f"{n_class}class" / f"{fold+1}fold"
        model_dir = "ML" / Path("model_container")/preprocessing_dir.split('_')[0] / f"{number_of_ch}ch"/f"ds_{ds}"/f"{n_class}class" / f"{fold+1}fold"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        callbacks_list = [EarlyStopping(monitor="val_loss", patience=3),
                        ModelCheckpoint(f"{model_dir}/target_{target_subject}_model{filename_change}.keras", save_best_only=True, monitor='val_loss', mode='min'),
                        CSVLogger(log_dir / f"global_{target_subject}_model{filename_change}.csv", append=False)]

        model.fit(X_train_combined, y_train_combined, epochs=100, batch_size=12, validation_data=(X_test, y_test), callbacks=callbacks_list)
        # model.fit(X_train_combined, y_train_combined, epochs=1, batch_size=500, validation_data=(X_test, y_test), callbacks=callbacks_list)

        # //////////Transfer Learning part//////////
        group_results = []
        # load data of target subject
        X_sstl, y_sstl, _ = load_data([target_subject_data_path], movement_types, ch_idx, n_class, number_of_ch)
        # Convert labels to categorical
        y_sstl = to_categorical(y_sstl, num_classes=n_class)

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

            # Transfer Learning
            history_sstl = general_model.fit(X_train_sstl, y_train_sstl, epochs=30, batch_size=12, validation_data=(X_val, y_val), callbacks=sstl_callbacks_list)
            # history_sstl = general_model.fit(X_train_sstl, y_train_sstl, epochs=1, batch_size=500, validation_data=(X_val, y_val), callbacks=sstl_callbacks_list)
            layer_name = "L8"
            intermediate_layer_model = Model(inputs=general_model.input,
                                    outputs=general_model.get_layer(layer_name).output)
            y_pred = intermediate_layer_model.predict(X_test_sstl)
            l8_outputs.append(y_pred)

    # リストをnumpy配列に変換
    l8_outputs = np.array(l8_outputs)

    # 平均を取る
    l8_outputs_mean = np.mean(l8_outputs, axis=0)

    # 次元削減（PCA）
    pca = PCA(n_components=10)
    reduced_data = pca.fit_transform(l8_outputs_mean)

    # k-meansクラスタリング
    kmeans = KMeans(n_clusters=5)  # クラスタ数は任意
    clusters = kmeans.fit_predict(reduced_data)

    # クラスタリングの評価
    sil_score = silhouette_score(reduced_data, clusters)
    print(f'Silhouette Score: {sil_score}')

    # 結果の可視化
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters)
    plt.title('PCA Reduced Data with K-means Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

message = "機械学習終了!!"
line_notify(message)
