from pathlib import Path
import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import BarChart, Reference

def get_subject_name(result_file):
    subject_name_row = result_file.iloc[1,:].dropna()
    new_subject_name_row = subject_name_row.reset_index(drop=True)
    subject_names = [new_subject_name_row[n].split("_")[1] for n in range(len(new_subject_name_row))]
    subject_names.sort()
    return subject_names, subject_name_row

def decide_index(result_file_path):
    parts = result_file_path.parts
    preprocessing = next((part for part in parts if part == "DWT" or part == "Envelope"), None)
    ch = next((part for part in parts if part == "64ch" or part == "28ch"), None)
    if preprocessing == "DWT":
        index_name = "新処理"
    elif preprocessing == "Envelope":
        index_name = "旧処理"
    return index_name, ch

preprocessing_type= "d" # the kind of preprocessing method{d(DWT), e(Envelope), b(BPF)}
number_of_chs = [64, 28] # How many channels to use (64, 38, 28, 19, 18, 12, 6)

current_dir = Path.cwd()
result_dir = current_dir / "ML" / "result"
result_file_paths = [result_file_path for result_file_path in list(result_dir.rglob("*.xlsx"))]

template_table_path = current_dir.parent / "result_table_create.xltx"
template_hist_path = current_dir.parent / "result_hist_create.xltx"

if template_table_path.exists() and template_hist_path.exists():
    for result_file_path in result_file_paths:
        index_name, ch = decide_index(result_file_path)

        result_file = pd.read_excel(result_file_path, index_col=None, header=None)
        subject_names, subject_name_row = get_subject_name(result_file)
        result_hist_col = subject_names.insert(0, ch)
        df = pd.DataFrame(index=["旧処理", "新処理"], columns=result_hist_col)
        result_file = result_file.iloc[:3,:]
        print(result_file)
        for subject_name in subject_names:
            for index in subject_name_row.index:
                if subject_name in result_file.iloc[1, index]:
                    acc = result_file.iloc[2, index+1]
                    df.loc[index_name, subject_name] = acc
                    print(df)



else:
    print("You mistake template file path. Please check the path.")
