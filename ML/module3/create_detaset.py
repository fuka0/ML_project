import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def select_electrode(number_of_ch=int):
    all_ch = ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6','Cp5', 'Cp3', 'Cp1',
            'Cpz', 'Cp2', 'Cp4', 'Cp6', 'Fp1', 'Fpz', 'Fp2', 'Af7', 'Af3', 'Afz', 'Af4','Af8', 'F7', 'F5', 'F3', 'F1','Fz',
            'F2', 'F4','F6', 'F8', 'Ft7', 'Ft8', 'T7', 'T8','T9', 'T10', 'Tp7', 'Tp8', 'P7', 'P5', 'P3', 'P1','Pz','P2',
            'P4','P6', 'P8', 'Po7','Po3', 'Poz', 'Po4', 'Po8', 'O1', 'Oz', 'O2', 'Iz']
    extracted_ch = []
    if number_of_ch == 64:
        ch_idx = np.arange(0, 64, 1).tolist()
        for i in ch_idx:
            extracted_ch.append(all_ch[i])

    elif number_of_ch == 38:
        ch_idx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 26, 28, 29, 31, 33, 35, 37, 40, 41, 42, 43, 46, 48, 50, 52, 54, 55, 57, 59, 60, 61, 62, 63]
        for i in ch_idx:
            extracted_ch.append(all_ch[i])

    elif number_of_ch == 19:
        ch_idx = [8, 10, 12, 21, 23, 29, 31, 33, 35, 37, 40, 41, 46, 48, 50, 52, 54, 60, 62]
        for i in ch_idx:
            extracted_ch.append(all_ch[i])

    elif number_of_ch == 6:
        ch_idx = [9, 11, 8, 12, 7, 13]
        for i in ch_idx:
            extracted_ch.append(all_ch[i])

    elif number_of_ch == 12:
        ch_idx = [2, 4, 1, 5, 9, 11, 8, 12, 16, 18, 15, 19]
        for i in ch_idx:
            extracted_ch.append(all_ch[i])

    elif number_of_ch == 18:
        ch_idx = [2, 4, 1, 5, 0, 6, 9, 11, 8, 12, 7, 13, 16, 18, 15, 19, 14, 20]
        for i in ch_idx:
            extracted_ch.append(all_ch[i])

    elif number_of_ch == 28:
        ch_idx = [2, 4, 1, 5, 0, 6, 9, 11, 8, 12, 7, 13, 16, 18, 15, 19, 14, 20, 10, 3, 17, 32, 34, 31, 35, 30, 36, 33]
        for i in ch_idx:
            extracted_ch.append(all_ch[i])
    return ch_idx, extracted_ch

def load_data(data_paths, movement_types, ch_idx, n_class, number_of_ch=int):
    scaler = StandardScaler()
    data_list = []
    label_list = []

    for data_path in data_paths:
        data_dict = np.load(data_path, allow_pickle=True).item()
        if n_class == 2:
            data = data_dict[movement_types[0]]["epoch_data"]
            label = data_dict[movement_types[0]]["labels"]
            reshaped_data = data.reshape(data.shape[0], -1)
            normalized_data = scaler.fit_transform(reshaped_data) # 標準化
            normalized_data = normalized_data.reshape(data.shape) # 元の形状に戻す

            # ラベルに基づいてデータを分ける
            for label_val in range(n_class):
                indices = np.where(label == label_val)[0]
                if number_of_ch == 64:
                    data_per_label = normalized_data[indices]
                else:
                    data_per_label = normalized_data[indices][:, :, ch_idx]
                labels_per_label = label[indices]

                data_list.append(data_per_label)
                label_list.append(labels_per_label)
        else:
            for movement_type in movement_types:
                data = data_dict[movement_type]["epoch_data"]
                label = data_dict[movement_type]["labels"]
                reshaped_data = data.reshape(data.shape[0], -1)
                normalized_data = scaler.fit_transform(reshaped_data) # 標準化
                normalized_data = normalized_data.reshape(data.shape) # 元の形状に戻す

                # ラベルに基づいてデータを分ける
                for label_val in range(n_class):
                    indices = np.where(label == label_val)[0]
                    if number_of_ch == 64:
                        data_per_label = normalized_data[indices]
                    else:
                        data_per_label = normalized_data[indices][:, :, ch_idx]
                    labels_per_label = label[indices]

                    data_list.append(data_per_label)
                    label_list.append(labels_per_label)
    # データを結合
        combined_data = np.concatenate(data_list, axis=0)
        combined_labels = np.concatenate(label_list, axis=0)
    return combined_data, combined_labels

def make_columns_subject(n_class, columns_li_binary, columns_li_multi):
    task_str_li = []
    columns = []
    for n in range(n_class):
        task_str_li.append(f"task{n}")
    if n_class > 2:
        for column in columns_li_multi:
            if "macro" in column:
                columns.append(column)
            else:
                for task_str in task_str_li:
                    columns.append(f"{task_str}_{column}")
    else:
        for column in columns_li_binary:
            for task_str in task_str_li:
                columns.append(f"{task_str}_{column}")
    return columns

def make_columns(n_class):
    if n_class == 2:
        columns = ["left_fist", "right_fist"]
    elif n_class == 3:
        columns = ["left_fist", "right_fist", "both_fists"]
    else:
        columns = ["left_fist", "right_fist", "both_fists", "both_feet"]
    return columns

def generate_noise(data, type_of_noise):
    if type_of_noise == "gauss":
        noise = np.random.normal(loc=0, scale=1, size=data.shape)
    elif type_of_noise == "white":
        noise = np.random.randn(*data.shape)
    return noise
