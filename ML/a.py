import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np

# 日本語フォントの設定
jp_font_path = "C:/Windows/Fonts/meiryo.ttc"
jp_font = fm.FontProperties(fname=jp_font_path)

# 既存のデータフレーム
data = {
    "S004": [69.17, 75.00],
    "S018": [57.22, 77.22],
    "S032": [68.33, 86.11],
    "S058": [59.44, 67.50],
    "S079": [59.44, 60.56]
}
df = pd.DataFrame(data, index=["旧処理", "新処理"])

# データフレームを整形
df = df.T.reset_index().melt(id_vars='index', var_name='処理', value_name='Accuracy')

# プロットの設定
plt.figure(figsize=(10, 6))
bar_width = 0.3  # バーの幅
spacing = 0.05  # 旧処理と新処理の間隔

# プロットの設定
ax = sns.barplot(x='index', y='Accuracy', hue='処理', data=df, palette=['blue', 'red'], dodge=False)

# 各バーの位置を調整
for i, bar in enumerate(ax.patches):
    if i % 2 == 0:  # 旧処理のバー
        bar.set_width(bar_width)
        bar.set_x(bar.get_x() - (bar_width + spacing) / 2)
    else:  # 新処理のバー
        bar.set_width(bar_width)
        bar.set_x(bar.get_x() + (bar_width + spacing) / 2)

# ラベルの追加
plt.xlabel('Subjects', fontproperties=jp_font)
plt.ylabel('Accuracy [%]', fontproperties=jp_font)
plt.title('Accuracy Comparison by Subject', fontproperties=jp_font)
plt.legend(title='処理', loc='upper right', prop=jp_font)

# 値ラベルの追加
def add_labels(ax):
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 10), textcoords='offset points',
                    fontproperties=jp_font)

ax = plt.gca()
add_labels(ax)

# グリッド線の追加
plt.grid(axis='y')

# プロットの表示
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 既存のデータフレーム
data = {
    "S004": [69.17, 75.00],
    "S018": [57.22, 77.22],
    "S032": [68.33, 86.11],
    "S058": [59.44, 67.50],
    "S079": [59.44, 60.56]
}
df = pd.DataFrame(data, index=["旧処理", "新処理"])

# プロットの設定
fig, ax = plt.subplots(figsize=(10, 6))
index = np.arange(len(df.columns))
bar_width = 0.35

# 棒グラフのプロット
bars1 = ax.bar(index, df.loc["旧処理"], bar_width, label="旧処理", color='b')
bars2 = ax.bar(index + bar_width, df.loc["新処理"], bar_width, label="新処理", color='r')

# ラベルの追加
ax.set_xlabel('Subjects')
ax.set_ylabel('Accuracy[%]')
ax.set_title('Accuracy Comparison by Subject')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(df.columns)
ax.legend()

# 値ラベルの追加
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)

# グリッド線の追加
ax.yaxis.grid(True)

# プロットの表示
plt.tight_layout()
plt.show()
