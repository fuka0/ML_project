from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from module3.create_detaset import select_electrode, load_data, make_columns, generate_noise
import os
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"

def generate_string(decompose_level, d_num):
    # decompose_levelから開始して、d_numの数だけ逆順にdetailを取得
    details = [f"d{i}" for i in range(decompose_level, decompose_level - d_num, -1)]
    # 文字列を"_"で結合
    return "_".join(details)

def Preprocessing(preprocessing_type):
    if preprocessing_type == "d":
        preprocessing_dir = "DWT"
    elif preprocessing_type == "e":
        preprocessing_dir = "Envelope"
    elif preprocessing_type == "b" :
        preprocessing_dir = "BPF"
    return preprocessing_dir
# ////////////////////////////////////////////////////////////////////////////////////////
n_class = 2 # 何クラス分類か(左手と右手なら2, 両手も含むなら3, 両足も含むなら4)
number_of_ch = 64 # 使用するチャンネル数(64, 38, 19, 8)
movement_types = ["left_right_fist", "fists_feet"] # 運動タイプを定義

extraction_section = True # 切り出し区間が安静時を含まないならTrue,含むならFalse
baseline_correction = True # ベースライン補正の有無
ext_sec = "move_only" if extraction_section else "rest_move"
baseline = "baseline_true" if baseline_correction else "baseline_false"
preprocessing_type= "d"

preprocessing_dir = Preprocessing(preprocessing_type)
ds = 3 # ダウンサンプリングの設定

d_num = 3 # 取得するdetailの個数(上から順に{D4,D3...})(2 or 3)
decompose_level = 5 # 分解レベル


ch_idx, extracted_ch = select_electrode(number_of_ch)
details_dir = generate_string(decompose_level, d_num)

current_dir = Path.cwd()
target_dir = current_dir / "ML" / "result" / preprocessing_dir / "SS-TL_acc" / f"{number_of_ch}ch" / f"ds_{ds}" / f"{n_class}class"
target_paths = list(Path.glob(target_dir, "*.txt"))
roc_dir = target_dir / "ROC"
learning_curve_dir = target_dir / "Learning_Curve"
os.makedirs(roc_dir, exist_ok=True)

fig, ax1 = plt.subplots(figsize=(10, 8))
ax1.grid()
for target_path in target_paths:
    fpr_loaded = []
    tpr_loaded = []
    thresholds_loaded = []
    subject_name = target_path.name.split("_")[3].split(".")[0]
    with open(target_path, "r") as file:
        auc_loaded = float(file.readline().split(":")[1])  # AUCを読み込む
        file.readline()  # 空行をスキップ
        next(file)  # ヘッダー行をスキップ
        for line in file:
            f, t, th = line.strip().split("\t")
            fpr_loaded.append(float(f))
            tpr_loaded.append(float(t))
            thresholds_loaded.append(float(th))

    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(fpr_loaded, tpr_loaded, label=f'{subject_name}_ROC curve (AUC = {auc_loaded:.2f})', linewidth=2)
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.tick_params(axis="both", labelsize=17)
    ax1.set_xlabel('False Positive Rate', fontsize=18)
    ax1.set_ylabel('True Positive Rate', fontsize=18)
    ax1.legend(loc="lower right", fontsize=15)
    ax1.set_title(f"ROC", fontsize=15)
    plt.savefig(roc_dir / f"{subject_name}_ROC.png")
    plt.clf()
    plt.close()


n_fold = 1
i = 0
logs_dir = current_dir / "ML" / "logs" / preprocessing_dir / f"{number_of_ch}ch" / f"ds_{ds}" / f"{n_class}class"
log_paths = list(logs_dir.rglob("*.csv"))

log_dict = {}
folds = set(log_path.parent.name for log_path in log_paths)
for fold in folds:
    log_path_list = [log_path for log_path in log_paths if log_path.parent.name == fold]
    log_dict[fold] = log_path_list

for fold, log_paths in log_dict.items():
    for log_path in log_paths:
        parts = log_path.parts
        model_name = log_path.name.split("_")[0]
        subject_name = log_path.name.split("_")[1]
        df = pd.read_csv(log_path)
        fig, ax2 = plt.subplots(1, 2, figsize=(13, 8))
        ax2[0].plot(df['loss'], "o-", label=f'{subject_name} Loss')
        ax2[0].plot(df['val_loss'], "o-", label=f'{subject_name} Validation Loss', color="orange")
        ax2[1].plot(df['accuracy'], "o-", label=f'{subject_name} Accuracy')
        ax2[1].plot(df['val_accuracy'], "o-", label=f'{subject_name} Validation Accuracy', color="orange")
        ax2[0].legend(fontsize=12)
        ax2[1].legend(fontsize=12)
        ax2[0].tick_params(axis="both", labelsize=15)
        ax2[1].tick_params(axis="both", labelsize=15)
        ax2[0].set_xlabel("Epoch", fontsize=18)
        ax2[1].set_xlabel("Epoch", fontsize=18)
        ax2[0].set_ylabel("Loss", fontsize=18)
        ax2[1].set_ylabel("Accuracy", fontsize=18)
        ax2[0].grid()
        ax2[1].grid()
        save_dir = learning_curve_dir / fold
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir / f"{model_name}_{subject_name}_LC.png")
        plt.clf()
        plt.close()
