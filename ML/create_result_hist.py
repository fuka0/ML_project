import os, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

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
n_class = 2 # How many class to classify (2 for left and right hands, 3 add for both hands, 4 add for both feet)
number_of_chs = [64, 28] # How many channels to use (64, 38, 28, 19, 18, 12, 6)
ds = 3 # down sampling rate(1/2, 1/3)

fileneme_change = "_2Q_result"

current_dir = Path.cwd()
result_dir = current_dir / "ML" / "result"
result_file_paths = [result_file_path for result_file_path in list(result_dir.rglob(f"*{fileneme_change}.xlsx"))] # get all result file paths

# make temporary dictionary to store the results
results_28ch = {"旧処理": {}, "新処理": {}}
results_64ch = {"旧処理": {}, "新処理": {}}

for result_file_path in result_file_paths:
    # print(result_file_path)
    parts = result_file_path.parts # disassemble the path of the result file
    ch = next((part for part in parts if part == "64ch" or part == "28ch"), None) # get the number of channels from the result file path
    if f"ds_{ds}" in parts and f"{n_class}class" in parts:
        index_name = decide_index(parts) # judge the preprocessing method

        result_file = pd.read_excel(result_file_path, index_col=None, header=None, engine="openpyxl") # load the result file
        subject_names, subject_name_row = get_subject_name(result_file) # get the subject names list

        result_file = result_file.iloc[:3,:] # clear the unnecessary rows
        for subject_name in subject_names:
            for index in subject_name_row.index:
                if subject_name in result_file.iloc[1, index]:
                    acc = result_file.iloc[2, index+1]
                    if ch == "64ch":
                        results_64ch[index_name][subject_name] = acc
                    elif ch == "28ch":
                        results_28ch[index_name][subject_name] = acc
        continue

df_64ch = create_dataframe_from_results(results_64ch)
df_28ch = create_dataframe_from_results(results_28ch)
jp_font_path = "C:/Windows/Fonts/meiryo.ttc"
en_font_path = "C:/Windows/Fonts/times.ttf"
jp_font = fm.FontProperties(fname=jp_font_path)
en_font = fm.FontProperties(fname=en_font_path)

save_dir_dict = {}
for ch in number_of_chs:
    save_dir = f"ML/result_hist/"
    save_dir_dict[f"{ch}ch"] = Path(save_dir) / f"{ch}ch" / f"ds_{ds}" / f"{n_class}class"
    os.makedirs(save_dir_dict[f"{ch}ch"], exist_ok=True)

save_path_64ch = save_dir_dict["64ch"] / "result_hist_64ch.png"
save_path_28ch = save_dir_dict["28ch"] / "result_hist_28ch.png"

# data formatting
df_64ch = df_64ch.T.reset_index().melt(id_vars="index", var_name="処理", value_name="Accuracy")
df_28ch = df_28ch.T.reset_index().melt(id_vars="index", var_name="処理", value_name="Accuracy")
bar_width = 0.36  # width of the bar

# plot settings(64ch)
plt.figure(figsize=(10, 6))
plt.ylim(0, 100)
label_fontsize = 19
value_label_fontsize = 18
legend_fontsize = 16

ax_64ch = sns.barplot(x="index", y="Accuracy", hue="処理", data=df_64ch, palette=["blue", "red"], dodge=True)

# bar position adjustment
for i, bar in enumerate(ax_64ch.patches):
    bar.set_width(bar_width)


# add labels
plt.xlabel("Subjects", fontproperties=en_font, fontsize=label_fontsize)
plt.ylabel("Accuracy [%]", fontproperties=en_font, fontsize=label_fontsize)

leg = plt.legend(loc="upper right", prop={"size": legend_fontsize}, frameon=False)


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
ax.set_xticklabels(df_64ch["index"].unique(), fontproperties=en_font, fontsize=label_fontsize)

yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels([f"{int(y)}" for y in yticks], fontproperties=en_font, fontsize=label_fontsize)

plt.tight_layout()
plt.savefig(save_path_64ch)
plt.close()
plt.clf()


# /////////////////plot of 28ch results/////////////////
plt.figure(figsize=(10, 6))
plt.ylim(0, 100)
ax_28ch = sns.barplot(x="index", y="Accuracy", hue="処理", data=df_28ch, palette=["blue", "red"], dodge=True)

# bar position adjustment
for i, bar in enumerate(ax_28ch.patches):
    bar.set_width(bar_width)


# add labels
plt.xlabel("Subjects", fontproperties=en_font, fontsize=label_fontsize)
plt.ylabel("Accuracy [%]", fontproperties=en_font, fontsize=label_fontsize)

leg = plt.legend(loc="upper right", prop={"size": legend_fontsize}, frameon=False)


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
ax.set_xticklabels(df_28ch["index"].unique(), fontproperties=en_font, fontsize=label_fontsize)

yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels([f"{int(y)}" for y in yticks], fontproperties=en_font, fontsize=label_fontsize)

plt.tight_layout()
plt.savefig(save_path_28ch)
plt.close()
plt.clf()
