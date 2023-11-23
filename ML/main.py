from pathlib import Path
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from keras.utils import to_categorical, plot_model
from module3.create_detaset import select_electrode, load_data, make_columns, generate_noise
from keras.models import load_model
from module2.create_model import one_dim_CNN_model, create_model_builder, line_notify, Bayesian_Opt
from sklearn.model_selection import StratifiedShuffleSplit
from skopt.space import Real, Categorical, Integer
import os
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import random
from collections import defaultdict
os.environ["OMP_NUM_THREADS"] = "1"


def generate_string(decompose_level, d_num):
    # decompose_levelから開始して、d_numの数だけ逆順にdetailを取得
    details = [f"d{i}" for i in range(decompose_level, decompose_level - d_num, -1)]
    # 文字列を"_"で結合
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
n_class = 2 # 何クラス分類か(左手と右手なら2, 両手も含むなら3, 両足も含むなら4)
number_of_chs = [64, 28] # 使用するチャンネル数(64, 38, 28, 19, 18, 12, 6)
movement_types = ["left_right_fist", "fists_feet"] # 運動タイプを定義

extraction_section = True # 切り出し区間が安静時を含まないならTrue,含むならFalse
baseline_correction = True # ベースライン補正の有無
ext_sec = "move_only" if extraction_section else "rest_move"
baseline = "baseline_true" if baseline_correction else "baseline_false"
preprocessing_type= "d" # d(DWT), e(Envelope), b(BPF)
ds = 2 # ダウンサンプリングの設定

d_num = 3 # 取得するdetailの個数(上から順に{D4,D3...})(2 or 3)
decompose_level = 5 # 分解レベル

for number_of_ch in number_of_chs:
    ch_idx, extracted_ch = select_electrode(number_of_ch)
    current_dir = Path.cwd()
    preprocessing_dir = Preprocessing(preprocessing_type)

    columns = make_columns(n_class)
    subjects_dir = [current_dir/preprocessing_dir/ext_sec/baseline/f"S{i+1:03}" for i in range(104)]
    subjects_name = [subject_dir.name for subject_dir in subjects_dir]
    random.seed(22)
    target_subjects = random.sample(subjects_name, 5)

    details_dir = generate_string(decompose_level, d_num)

    for target_subject in target_subjects:
        if target_subject == "S018" or target_subject == "S032" or target_subject == "S004":
            target_subject_index = target_subjects.index(target_subject)
            if n_class > 2:
                df = pd.DataFrame(index=[f"target_{target_subject}", "ACC", "PRE", "REC", "F1", "macro_REC", "macro_PRE", "macro_F1"], columns=columns)
            else:
                df = pd.DataFrame(index=[f"target_{target_subject}", "ACC", "PRE", "REC", "F1"], columns=columns)

            if preprocessing_dir == "DWT_data":
                data_paths = list((current_dir / "ML" / "ref_data" / preprocessing_dir / f"decomposition_level{decompose_level}" / details_dir / ext_sec / f"ds_{ds}" ).glob(f"S*/*.npy"))
            elif preprocessing_dir == "Envelope_data":
                data_paths = list((current_dir / "ML" / "ref_data" / preprocessing_dir / ext_sec / f"ds_{ds}" ).glob(f"S*/*.npy"))
            elif preprocessing_dir == "BPF_data":
                data_paths = list((current_dir / "ML" / "ref_data" / preprocessing_dir / ext_sec / f"ds_{ds}" ).glob(f"S*/*.npy"))

            for data_path in data_paths:
                if data_path.parent.name == target_subject:
                    # ターゲットとなる被験者のデータpath
                    target_subject_data_path = data_path
                    # data_pathからtarget_subjectのpathを除外
                    data_paths.remove(data_path)

            X, y = load_data(data_paths, movement_types, ch_idx, n_class, number_of_ch)
            left_fist_idx = np.where(y == 0)
            right_fist_idx = np.where(y == 1)
            both_fists = np.where(y == 2)
            both_feet = np.where(y == 3)

            sss = StratifiedShuffleSplit(n_splits=5,random_state=17,test_size=0.2)

            # ベイズ最適化
            # best_params = Bayesian_Opt(X, y, n_class, n_iter=150, cv=5)

            # Lists to store the accuracy of each fold
            train_acc_list = []
            val_acc_list = []

            # 混同行列を格納するためのリスト
            conf_matrices = []
            # recall, precision, F-valueを格納するためのリスト
            recalls, precisions, f_values = [], [], []

            # Split the dataset into training and testing sets for each fold
            for train, test in sss.split(X, y):
                X_train, X_test = X[train], X[test]
                y_train, y_test = y[train], y[test]

                # ノイズ生成
                noise = generate_noise(X_train, "gauss")
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
                # modelを定義
                model = one_dim_CNN_model(input_shape, n_class, optimizer='adam', learning_rate=0.001)
                # plot_model(model, to_file="global_model.png", show_shapes=True)
                if preprocessing_dir == "DWT_data":
                    log_dir = current_dir / "ML" / "logs" / preprocessing_dir.split("_")[0] / f"decomposition_level{decompose_level}" / details_dir / ext_sec / f"{number_of_ch}ch" / f"ds_{ds}" / f"{n_class}class"
                    model_dir = "ML" / Path("model_container")/preprocessing_dir.split('_')[0]/f"decomposition_level{decompose_level}" / details_dir /ext_sec/f"{number_of_ch}ch"/f"ds_{ds}"/f"{n_class}class"
                    os.makedirs(log_dir, exist_ok=True)
                    os.makedirs(model_dir, exist_ok=True)
                elif preprocessing_dir == "Envelope_data" or preprocessing_dir == "BPF_data":
                    log_dir = current_dir / "ML" / "logs" / preprocessing_dir.split("_")[0] / ext_sec / f"{number_of_ch}ch" / f"ds_{ds}" / f"{n_class}class"
                    model_dir = "ML" / Path("model_container")/preprocessing_dir.split('_')[0] / ext_sec/f"{number_of_ch}ch"/f"ds_{ds}"/f"{n_class}class"
                    os.makedirs(log_dir, exist_ok=True)
                    os.makedirs(model_dir, exist_ok=True)

                callbacks_list = [EarlyStopping(monitor="val_loss", patience=4),
                                ModelCheckpoint(f"{model_dir}/target_{target_subject}_model.keras", save_best_only=True, monitor='val_loss')]

                history = model.fit(X_train_combined, y_train_combined, epochs=200, batch_size=12, validation_data=(X_test, y_test), callbacks=callbacks_list)
        # 転移学習part//////////////////////////////////
            group_results = []
            sstl_train_acc_list = []
            sstl_val_acc_list = []

            # データのロード
            X_sstl, y_sstl = load_data([target_subject_data_path], movement_types, ch_idx, n_class, number_of_ch)
            # ラベルをカテゴリカルに変換
            y_sstl = to_categorical(y_sstl, num_classes=n_class)
            print(y_sstl.shape)
            X_sstl = X_sstl[:70,:,:]
            y_sstl = y_sstl[:70,:]
            print(X_sstl.shape, y_sstl.shape)
            sss = StratifiedShuffleSplit(n_splits=5,random_state=22,test_size=0.2)

            # 混同行列を格納するためのリスト
            conf_matrices = []
            # recall, precision, F-valueを格納するためのリスト
            recalls, precisions, f_values = [], [], []
            # 各分割での訓練と検証
            for train, test in sss.split(X_sstl, y_sstl):
                # グローバルモデルのロード
                global_model = load_model(f"{model_dir}/target_{target_subject}_model.keras")
                for layer in global_model.layers:
                    if layer.name == "L4":
                        layer.trainable = True
                        break
                    layer.trainable = False

                # # 確認のためにモデルのtrainable属性を出力
                # for layer in global_model.layers:
                #     print(layer.name, layer.trainable)
                # plot_model(global_model, to_file="SS-TL_model.png", show_shapes=True)
                X_train_sstl, X_test_sstl = X_sstl[train], X_sstl[test]
                y_train_sstl, y_test_sstl = y_sstl[train], y_sstl[test]

                sstl_callbacks_list = [EarlyStopping(monitor="val_loss", patience=4),
                                    CSVLogger(log_dir / f"target_{target_subject}_model.csv",append=True)]

                # SS-TL学習
                history_sstl = global_model.fit(X_train_sstl, y_train_sstl, epochs=30, batch_size=12, validation_data=(X_test_sstl, y_test_sstl), callbacks=sstl_callbacks_list)
                y_pred = global_model.predict(X_test_sstl)

                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_test_sstl, axis=1)

                # ROC曲線
                y_true_prob = y_test_sstl[:, 1]
                y_pred_prob = y_pred[:, 1]
                fpr, tpr, thresholds = roc_curve(y_true_prob, y_pred_prob)
                # AUCを計算
                auc = roc_auc_score(y_true_prob, y_pred_prob)

                # 混同行列を計算
                conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
                conf_matrices.append(conf_matrix)

            # 混同行列の平均を計算
            average_conf_matrix = np.mean(conf_matrices, axis=0)
            n_classes = average_conf_matrix.shape[0]
            # マクロ平均用の入れ物
            macro_recall = 0
            macro_precision = 0
            macro_f_value = 0
            for i in range(n_classes):
                TP = average_conf_matrix[i, i]
                FP = sum(average_conf_matrix[:, i]) - TP
                FN = sum(average_conf_matrix[i, :]) - TP
                TN = sum(sum(average_conf_matrix)) - TP - FP - FN
                # 評価指標を求める
                accuracy = sum(average_conf_matrix.diagonal()) / sum(sum(average_conf_matrix))
                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
                f_value = 2 * precision * recall / (precision + recall)

                # マクロ平均の計算
                macro_recall += recall
                macro_precision += precision
                macro_f_value += f_value

                group_results.append({f"task{i}_ACC": f"{accuracy * 100:.2f}", f"task{i}_PRE": f"{precision * 100:.2f}",
                                    f"task{i}_REC": f"{recall * 100:.2f}", f"task{i}_F1": f"{f_value * 100:.2f}"})
            # 多クラス分類のときマクロ平均を求める
            if n_classes > 2:
                macro_recall = macro_recall / n_classes
                macro_precision = macro_precision / n_classes
                macro_f_value = macro_f_value / n_classes

                group_results.append({f"macro_PRE": f"{macro_precision * 100:.2f}", f"macro_REC": f"{macro_recall * 100:.2f}",
                                    f"macro_F1": f"{macro_f_value * 100:.2f}"})
            # グループの評価指標を加算
            result = defaultdict(float)
            for group_result in group_results:
                for k, v in group_result.items():
                    result[k] += float(v)
            result = dict(result)

            movetype = ["left_fist", "right_fist", "both_fists", "both_feet"]
            # グループの評価指標の平均を求める
            df_value = defaultdict(float)
            subjects_num = len([target_subject_data_path])
            for key, value in result.items():
                if "macro" in key:
                    pass
                else:
                    key = f"{movetype[int(key.split('_')[0][-1])]}_{key.split('_')[1]}"
                df_value[key] = float(f"{value / subjects_num:.2f}")
            df_value = dict(df_value)

            # グループの評価指標の平均をDataFrameに入力
            for column in df_value.keys():
                if "macro" in column:
                    df.loc[column] = df_value[column]
                else:
                    split_column = column.split('_')
                    new_column = f"{split_column[0]}_{split_column[1]}"
                    df.loc[split_column[2], new_column] = df_value[column]
            # グループ平均の結果を出力
            if preprocessing_dir == "DWT_data":
                save_dir = current_dir / "result"/ preprocessing_dir.split("_")[0] / f"decomposition_level{decompose_level}" / details_dir / ext_sec / "SS-TL_acc" / f"{number_of_ch}ch" / f"ds_{ds}" / f"{n_class}class"
                os.makedirs(save_dir, exist_ok=True)
            elif preprocessing_dir == "Envelope_data" or preprocessing_dir == "BPF_data":
                save_dir = current_dir / "result"/ preprocessing_dir.split("_")[0] / ext_sec / "SS-TL_acc" / f"{number_of_ch}ch" / f"ds_{ds}" / f"{n_class}class"
                os.makedirs(save_dir, exist_ok=True)
            # テキストファイルとして保存
            with open(save_dir / f"roc_auc_data_{target_subject}.txt", "w") as file:
                file.write(f"AUC: {auc}\n\n")  # AUCを書き込む
                file.write("FPR\tTPR\tThresholds\n")  # タブ区切りのヘッダーを書き込む
                for f, t, th in zip(fpr, tpr, thresholds):
                    file.write(f"{f}\t{t}\t{th}\n")

            save_path = save_dir / f"ave_evalute.xlsx"
            sheet_name = "SS-TL_model_evaluate"
            if target_subject_index == 0:
                df.to_excel(save_path, sheet_name=sheet_name)
            else:
                with pd.ExcelWriter(save_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                    df.to_excel(writer, startrow=0, startcol=target_subject_index*(len(columns)+2), sheet_name=sheet_name)

message = "機械学習終了!!"
line_notify(message)
