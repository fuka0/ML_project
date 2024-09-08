import os, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

def get_subject_name(result_file):
    subject_name_row = result_file.iloc[1,:].dropna()
    new_subject_name_row = subject_name_row.reset_index(drop=True)
    subject_names = [new_subject_name_row[n].split("_")[1] for n in range(len(new_subject_name_row))]
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

# 値ラベルの追加
def add_labels(ax):
    for p in ax.patches:
        ax.annotate(format(p.get_height(), ".2f"),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = "center", va = "center",
                    xytext = (0, 10), textcoords = "offset points",
                    fontproperties=en_font,
                    fontsize=value_label_fontsize)
# ////////////////////////////////////////////////////////////////////////////////////////
number_of_chs = [64] # How many channels to use
subject = "S004" # S004, S018, S032, S058, S079

if subject == "S004":
    # S004
    results = {
        "2class classification(L, R)": {"0": 91.67, "50": 93.42, "100": 94.74, "150": 97.69, "200": 97.76, "250":98.47, "300": 98.26},
        "3class classification(L, R, B)": {"0": 74.63, "50": 87.28, "100": 91.78, "150": 93.97, "200": 96.63, "250":96.89, "300": 97.51},
        "4class classification(L, R, B, F)": {"0": 65.69, "50": 85.39, "100": 90.39, "150": 93.43, "200": 94.62, "250":96.27, "300": 96.43}
    }
elif subject == "S018":
    # S018
    results = {
        "2class classification(L, R)": {"0": 86.67, "50": 93.68, "100": 98.02, "150": 96.99, "200": 98.32, "250":98.52, "300": 98.01},
        "3class classification(L, R, B)": {"0": 69.29, "50": 85.34, "100": 89.89, "150": 93.94, "200": 94.12, "250":95.03, "300": 96.42},
        "4class classification(L, R, B, F)": {"0": 58.89, "50": 81.64, "100": 88.79, "150": 91.25, "200": 93.65, "250":94.81, "300": 94.38}
    }
elif subject == "S032":
    # S032
    results = {
        "2class classification(L, R)": {"0": 95.00, "50": 97.37, "100": 99.22, "150": 98.72, "200": 99.54, "250":99.19, "300": 99.35},
        "3class classification(L, R, B)": {"0": 75.00, "50": 88.51, "100": 94.89, "150": 96.15, "200": 97.52, "250":97.06, "300": 96.55},
        "4class classification(L, R, B, F)": {"0": 73.33, "50": 85.53, "100": 91.94, "150": 94.49, "200": 94.77, "250":95.89, "300": 96.68}
    }
elif subject == "S058":
    # S058
    results = {
        "2class classification(L, R)": {"0": 75.56, "50": 92.11, "100": 95.26, "150": 95.96, "200": 97.35, "250":97.75, "300": 98.48},
        "3class classification(L, R, B)": {"0": 68.75, "50": 82.76, "100": 88.64, "150": 94.03, "200": 94.02, "250":94.89, "300": 95.55},
        "4class classification(L, R, B, F)": {"0": 56.39, "50": 82.96, "100": 88.32, "150": 91.09, "200": 93.06, "250":94.45, "300": 94.91}
    }
elif subject == "S079":
    # S079
    results = {
        "2class classification(L, R)": {"0": 67.78, "50": 87.76, "100": 91.03, "150": 95.13, "200": 95.71, "250":96.74, "300": 97.43},
        "3class classification(L, R, B)": {"0": 43.89, "50": 70.09, "100": 82.07, "150": 88.03, "200": 91.46, "250":94.32, "300": 95.36},
        "4class classification(L, R, B, F)": {"0": 34.31, "50": 67.37, "100": 79.96, "150": 85.58, "200": 89.52, "250":93.41, "300": 93.71}
    }

df = create_dataframe_from_results(results)
jp_font_path = "C:/Windows/Fonts/meiryo.ttc"
en_font_path = "C:/Windows/Fonts/times.ttf"
jp_font = fm.FontProperties(fname=jp_font_path)
en_font = fm.FontProperties(fname=en_font_path)

save_dir_dict = {}
for ch in number_of_chs:
    save_dir = f"ML/result_artificial_plot/"
    save_dir_dict[f"{ch}ch"] = Path(save_dir) / "multi_stream_1d_5-7-11-13" / f"{ch}ch"  / "D5-D4-D3"
    os.makedirs(save_dir_dict[f"{ch}ch"], exist_ok=True)

save_path_64ch = save_dir_dict[f"{ch}ch"] / f"result_plot_{subject}.png"

# data formatting
df = df.T.reset_index().melt(id_vars="index", var_name="Number of classification class", value_name="Accuracy")
bar_width = 0.23  # width of the bar

# plot settings(64ch)
plt.figure(figsize=(10, 6))
plt.ylim(20, 100)
label_fontsize = 18
value_label_fontsize = 15
legend_fontsize = 15

ax_64ch = sns.lineplot(x="index", y="Accuracy", hue="Number of classification class", data=df, palette=["steelblue", "tomato", "forestgreen"],
                    markers=["o", "s", "D"], dashes=False, style="Number of classification class", legend="full")

# add labels
plt.xlabel("Amount of artificial EEG data", fontproperties=en_font, fontsize=label_fontsize)
plt.ylabel("Accuracy [%]", fontproperties=en_font, fontsize=label_fontsize)

leg = plt.legend(loc="lower right", prop={"size": legend_fontsize}, frameon=True)

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

# x-axis label position adjustment
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_xticklabels(df["index"].unique(), fontproperties=en_font, fontsize=label_fontsize)

yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels([f"{int(y)}" for y in yticks], fontproperties=en_font, fontsize=label_fontsize)

plt.tight_layout()
# plt.show()
plt.savefig(save_path_64ch)
plt.clf()
plt.close()
