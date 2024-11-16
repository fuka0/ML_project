import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

def get_subject_name(result_file):
    subject_name_row = result_file.iloc[0, :].dropna()
    new_subject_name_row = subject_name_row.reset_index(drop=True)
    subject_names = []
    for n in range(len(new_subject_name_row)):
        if isinstance(new_subject_name_row[n], str):
            split_parts = new_subject_name_row[n].split("_")
            if len(split_parts) > 1:
                subject_names.append(split_parts[1])
    subject_names.sort()
    return subject_names, subject_name_row

def decide_index(parts):
    preprocessing = next((part for part in parts if part == "DWT" or part == "Envelope"), None)
    if preprocessing == "DWT":
        index_name = "新処理"
    elif preprocessing == "Envelope":
        index_name = "旧処理"
    return index_name

def create_dataframe_from_results(results):
    df = pd.DataFrame(results).T
    return df


def add_labels(ax):
    for p in ax.patches:
        ax.annotate(format(p.get_height(), ".2f"),
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.05),  # ラベルを少し上に移動
                    ha="center", va="bottom",
                    fontsize=value_label_fontsize,
                    fontproperties=en_font,
                    rotation=23)  # ラベルを少し斜めに配置

# 値ラベルの追加
# def add_labels(ax):
#     for p in ax.patches:
#         ax.annotate(format(p.get_height(), ".2f"),
#                     (p.get_x() + p.get_width() / 2., p.get_height()),
#                     ha = "center", va = "center",
#                     xytext = (0, 10), textcoords = "offset points",
#                     fontproperties=en_font,
#                     fontsize=value_label_fontsize)

def result_indexses(num_class, subj):
    index_mapping = {
        2: {"S004": 9, "S018": 1, "S032": 5, "S058": 17, "S079": 13},
        3: {"S004": 11, "S018": 1, "S032": 6, "S058": 21, "S079": 16},
        4: {"S004": 13, "S018": 1, "S032": 7, "S058": 25, "S079": 19}
    }
    return index_mapping.get(num_class, {}).get(subj, None)

def process_result_files(file_paths, ds, num_class, results):
    for file_path in file_paths:
        result_file = pd.read_excel(file_path)
        subject, _ = get_subject_name(result_file)
        acc_rows = result_file[result_file.iloc[:, 0] == 'ACC']

        if acc_rows.empty:
            continue

        for _, acc_row in acc_rows.iterrows():
            for _, subj in enumerate(subject):
                # Fill results only for the appropriate classification type
                try:
                    acc_value = acc_row.iloc[result_indexses(num_class, subj)] # Only take the first 'left_fist' column since all are the same
                    results[f"ds=1/{ds}"][subj] = "{:.2f}".format(float(acc_value))
                except (IndexError, ValueError):
                    continue
    return results
# ////////////////////////////////////////////////////////////////////////////////////////
current_file_path = Path(__file__).resolve() # get the current file path
target_name = "Fuka" # target parent directory name (You can change this name flexibly)
target_path = None # initialize the target path

for parent in current_file_path.parents:
    if parent.name == target_name:
        target_path = parent
        break

if target_path: # if you find the target directory, change the current directory to the target directory
    os.chdir(target_path)
else:
    print(f"Could not find the target directory: {target_name}")
    sys.exit()

target_result_dir = r"OneDrive - 東京都市大学 Tokyo City University/研究結果" # target directory name (You can change this name flexibly)
os.chdir(target_result_dir) # change the current directory to the target directory
current_dir = Path.cwd() # get the current directory
base_dir = current_dir

subjects = ["S004", "S018", "S032", "S058", "S079"] # S004, S018, S032, S058, S079

# difine directory path settings////////////////////////////
analyzed_interval = "3.0s" # 0.5s, 1.0s, 3.0s
preprocessing = "DWT"
number_of_ch = 64 # How many channels to use
task_type = 0 # actual_imagine = 0, actual = 1, imagine = 2
task_name_list = ["actual_imagine", "actual", "imagine"]
ds_list = [1, 2, 3] # down sampling rate {1(1/1) | 2(1/2) | 3(1/3)}
num_classes = [2, 3, 4]

results = {
            "ds=1/1": {"S004": None, "S018": None, "S032": None, "S058": None, "S079": None},
            "ds=1/2": {"S004": None, "S018": None, "S032": None, "S058": None, "S079": None},
            "ds=1/3": {"S004": None, "S018": None, "S032": None, "S058": None, "S079": None}
        }
for num_class in num_classes:
    for ds in ds_list:
        target_dir = f"analysis_interval_{analyzed_interval}/result/{preprocessing}/SS-TL_acc/{number_of_ch}ch/{task_name_list[task_type]}/D5-D4-D3/ds_{ds}/{num_class}class"
        os.chdir(target_dir)
        result_dir = Path.cwd()
        target_file_list = list(result_dir.glob("ave_*_L4.xlsx"))
        total_results = process_result_files(target_file_list, ds, num_class, results)
        os.chdir(base_dir) # change the current directory to the base directory

    os.chdir(current_file_path.parent)
    df = create_dataframe_from_results(total_results)
    jp_font_path = "C:/Windows/Fonts/meiryo.ttc"
    en_font_path = "C:/Windows/Fonts/times.ttf"
    jp_font = fm.FontProperties(fname=jp_font_path)
    en_font = fm.FontProperties(fname=en_font_path)

    save_dir = f"result_artificial_hist/analysis_interval_{analyzed_interval}/{number_of_ch}ch/{num_class}class"
    os.makedirs(save_dir, exist_ok=True)
    save_file_path = f"{save_dir}/result_hist.png"

    # data formatting
    df = df.T.reset_index().melt(id_vars="index", var_name="dawn sampling level", value_name="Accuracy")
    df["Accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")
    bar_width = 0.20  # width of the bar
    x = np.arange(len(subjects))

    # plot settings
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 100)
    label_fontsize = 30
    value_label_fontsize = 17
    legend_fontsize = 16

    ax = sns.barplot(x="index", y="Accuracy", palette=["b", "r", "g"], hue="dawn sampling level", data=df, dodge=True)

    # bar position adjustment
    for i, bar in enumerate(ax.patches):
        bar.set_width(bar_width)


    # add labels
    plt.xlabel("Subjects", fontproperties=en_font, fontsize=label_fontsize)
    plt.ylabel("Accuracy [%]", fontproperties=en_font, fontsize=label_fontsize)

    leg = plt.legend(loc="upper right", prop={"size": legend_fontsize}, frameon=True, bbox_to_anchor=(1.0, 1.02))


    # legend font properties
    for text in leg.get_texts():
        text.set_fontproperties(jp_font)
        text.set_fontsize(legend_fontsize)
    leg.get_title().set_fontproperties(jp_font)
    leg.get_title().set_fontsize(legend_fontsize)

    ax = plt.gca()
    add_labels(ax)

    # add grid lines
    plt.grid(axis="y", color="black")
    ax.set_axisbelow(True)

    # x-axis label position adjustment
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(df["index"].unique(), fontproperties=en_font, fontsize=label_fontsize)

    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(y)}" for y in yticks], fontproperties=en_font, fontsize=label_fontsize)

    plt.tight_layout()
    plt.savefig(save_file_path)
    plt.close()
    plt.clf()
    # plt.show()
    os.chdir(base_dir)
