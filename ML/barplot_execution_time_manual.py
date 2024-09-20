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
n_class = 4 # How many class to classify (2 for left and right hands, 3 add for both hands, 4 add for both feet)
number_of_chs = [64] # How many channels to use (64, 38, 28, 19, 18, 12, 6)

results_S032 = {
    "2class classification(L, R)": {"0": 85.29, "50": 61.24, "100": 89.49, "150": 104.68, "200": 92.57, "250": 102.16, "300": 98.13},
    "3class classification(L, R, B)": {"0": 128.94, "50": 132.41, "100": 147.58, "150": 140.83, "200": 133.60, "250": 141.77, "300": 163.72},
    "4class classification(L, R, B, F)": {"0": 170.13, "50": 164.12, "100": 215.26, "150": 189.24, "200": 202.30, "250": 210.75, "300": 222.90}
}

results_S079 = {
    "2class classification(L, R)": {"0": 74.77, "50": 70.69, "100": 76.03, "150": 90.31, "200": 88.63, "250": 91.97, "300": 94.94},
    "3class classification(L, R, B)": {"0": 131.35, "50": 132.95, "100": 134.37, "150": 141.06, "200": 162.67, "250": 155.95, "300": 163.46},
    "4class classification(L, R, B, F)": {"0": 169.14, "50": 185.35, "100": 193.60, "150": 199.95, "200": 209.87, "250": 199.50, "300": 214.56}
}

df = create_dataframe_from_results(results_S032)
jp_font_path = "C:/Windows/Fonts/meiryo.ttc"
en_font_path = "C:/Windows/Fonts/times.ttf"
jp_font = fm.FontProperties(fname=jp_font_path)
en_font = fm.FontProperties(fname=en_font_path)

save_dir_dict = {}
for ch in number_of_chs:
    save_dir = f"ML/result_hist/"
    save_dir_dict[f"{ch}ch"] = Path(save_dir) / "multi_stream_1d_5-7-11-13" / f"{ch}ch"  / "D5-D4-D3" / f"{n_class}class"
    os.makedirs(save_dir_dict[f"{ch}ch"], exist_ok=True)

save_path_64ch = save_dir_dict[f"{ch}ch"] / f"result_hist_{n_class}.png"

# data formatting
df = df.T.reset_index().melt(id_vars="index", var_name="execute_time", value_name="time")
bar_width = 0.23  # width of the bar

# plot settings(64ch)
plt.figure(figsize=(10, 6))
plt.ylim(50, 230)
label_fontsize = 18
value_label_fontsize = 15
legend_fontsize = 15

ax_64ch = sns.barplot(x="index", y="time", hue="execute_time", data=df, dodge=True)

# bar position adjustment
for i, bar in enumerate(ax_64ch.patches):
    bar.set_width(bar_width)


# add labels
plt.xlabel("Amount of artificial EEG data", fontproperties=en_font, fontsize=label_fontsize)
plt.ylabel("execution time [s]", fontproperties=en_font, fontsize=label_fontsize)

leg = plt.legend(loc="upper left", prop={"size": legend_fontsize}, frameon=False)


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
plt.show()
# plt.savefig(save_path_64ch)
# plt.clf()
# plt.close()
