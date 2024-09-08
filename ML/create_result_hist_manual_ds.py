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

# results = {
#     "ds=1/1": {"S004": 27.50, "S018": 44.17, "S032": 29.58, "S058": 32.92, "S079": 27.64},
#     "ds=1/2": {"S004": 54.03, "S018": 48.47, "S032": 60.28, "S058": 48.06, "S079": 32.36},
#     "ds=1/3": {"S004": 65.69, "S018": 58.89, "S032": 73.33, "S058": 56.39, "S079": 34.31},
# }

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
    save_dir = f"ML/result_hist/"
    save_dir_dict[f"{ch}ch"] = Path(save_dir) / "multi_stream_1d_5-7-11-13" / f"{ch}ch"  / "D5-D4-D3" / f"{n_class}class"
    os.makedirs(save_dir_dict[f"{ch}ch"], exist_ok=True)

save_path_64ch = save_dir_dict[f"{ch}ch"] / f"result_hist_{n_class}.png"

# data formatting
df = df.T.reset_index().melt(id_vars="index", var_name="処理", value_name="Accuracy")
bar_width = 0.23  # width of the bar

# plot settings(64ch)
plt.figure(figsize=(10, 6))
plt.ylim(0, 100)
label_fontsize = 18
value_label_fontsize = 15
legend_fontsize = 15

ax_64ch = sns.barplot(x="index", y="Accuracy", hue="処理", data=df, palette=["blue", "red", "green"], dodge=True)

# bar position adjustment
for i, bar in enumerate(ax_64ch.patches):
    bar.set_width(bar_width)


# add labels
plt.xlabel("Subjects", fontproperties=en_font, fontsize=label_fontsize)
# plt.ylabel("Accuracy [%]", fontproperties=en_font, fontsize=label_fontsize)
plt.ylabel("Accuracy [%]", fontproperties=en_font, fontsize=label_fontsize)

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
