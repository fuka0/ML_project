from pathlib import Path
import numpy as np
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, plot_model
from module3.create_detaset import select_electrode, load_data, make_columns, generate_noise
from module2.create_model import one_dim_CNN_model, line_notify
import os

from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
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

filename_change = ""

d_num = 3 # Number of detail to use(D4,D3...)(2 or 3)
decompose_level = 5 # Number of decomposition level

reduce_data = False # data reduction(True or False)
num_samples = 90  # Number of samples to use when reducing data(default=90)

ch_idx, extracted_ch = select_electrode(number_of_ch)
current_dir = Path.cwd()
preprocessing_dir = Preprocessing(preprocessing_type)

columns = make_columns(n_class)
subjects_dir = [current_dir/preprocessing_dir/ext_sec/baseline/f"S{i+1:03}" for i in range(104)]
subjects_name = [subject_dir.name for subject_dir in subjects_dir]

details_dir = generate_string(decompose_level, d_num)
l8_outputs = []
for target_subject in subjects_name:
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

    X_train, y_train, subject_ids = load_data(data_paths, movement_types, ch_idx, n_class, number_of_ch)
    X_test, y_test, _ = load_data([target_subject_data_path], movement_types, ch_idx, n_class, number_of_ch)
    
    left_fist_idx = np.where(y_train == 0)
    right_fist_idx = np.where(y_train == 1)
    both_fists = np.where(y_train == 2)
    both_feet = np.where(y_train == 3)

    # Lists to store the accuracy of each fold
    train_acc_list = []
    val_acc_list = []

    # List to store confusion matrices
    conf_matrices = []

    # List to store recall, precision, F-value
    recalls, precisions, f_values = [], [], []
    conf_matrices_dict = {}

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

    callbacks_list = [EarlyStopping(monitor="val_loss", patience=3)]

    model.fit(X_train_combined, y_train_combined, epochs=100, batch_size=12, validation_split=0.2, callbacks=callbacks_list)
    
    layer_name = "L8"
    intermediate_layer_model = Model(inputs=model.input,
                                    outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(X_test)
    l8_outputs.append(intermediate_output.mean(axis=0))

similarities = cosine_similarity(l8_outputs)

# KMeansクラスタリングを使用してグルーピング
n_clusters = 3  # 適切なクラスタ数を選択
kmeans = KMeans(n_clusters=n_clusters, random_state=22)
clusters = kmeans.fit_predict(similarities)

# クラスターごとの被験者を表示
for cluster_id in range(n_clusters):
    cluster_indices = np.where(clusters == cluster_id)[0]
    modified_cluster_indices = [f"S{str(cluster_subject + 1).zfill(3)}" for cluster_subject in cluster_indices]
        
    print(f"Cluster {cluster_id}: {modified_cluster_indices}")

top_k = 15
similar_subjects_indices = np.argsort(similarities)[-top_k:]
print(similar_subjects_indices)

message = "機械学習終了!!"
line_notify(message)
