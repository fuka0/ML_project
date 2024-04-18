import os
import mne
import numpy as np
from pathlib import Path

extraction_section = True # 切り出し区間が安静時を含まないならTrue,含むならFalse
baseline_correction = True # ベースライン補正の有無
ext_sec = "move_only" if extraction_section else "rest_move"
baseline = "baseline_true" if baseline_correction else "baseline_false"

current_path = Path.cwd()
base_path = current_path / "ML" /"EEG_dataset" / "files"
tmp_save_path = current_path / "ML" / "ref_data" / "ML_data" / ext_sec

# S088, S089, S092, S100, S104を除外
subject_dirs = [f"S{i:03d}" for i in range(1, 110) if i not in [88, 89, 92, 100, 104]]

left_right_fist = [3, 4, 7, 8, 11, 12]
both_fists_feet = [5, 6, 9, 10, 13, 14]

type_of_movement = "fist" # fist or both_fists_feet

samplefreq = 160 # サンプリング周波数
cut_out_time = 3 # 切り出し秒数

file_paths = []
for idx, subject_dir in enumerate(subject_dirs):  # インデックスを使用して保存先のディレクトリ名を連番に調整
    subject_path = base_path / subject_dir
    if type_of_movement == "fist":
        patterns = [subject_path / f'{subject_dir}R{i:02d}.edf' for i in left_right_fist]
    elif type_of_movement == "both_fists_feet":
        patterns = [subject_path / f'{subject_dir}R{i:02d}.edf' for i in both_fists_feet]
    file_paths.append(patterns)

num = 0
for file_path in file_paths:
    subject_epoch_data = []
    subject_labels = []
    for file in file_path:
        raw = mne.io.read_raw_edf(file,preload=True)
        # for ann in raw.annotations:
        #     print('Onset:', ann['onset'])
        #     print('Duration:', ann['duration'])
        #     print('Description:', ann['description'])
        raw.filter(l_freq=0.5,h_freq=45,verbose=False)
        events, event_id = mne.events_from_annotations(raw)
        if type_of_movement == "fist":
            event_id = {'left_fist': event_id['T1'], 'right_fist': event_id['T2']}
        else:
            event_id = {'both_fist': event_id['T1'], 'both_feet': event_id['T2']}

        if baseline_correction and extraction_section:
            epochs = mne.Epochs(raw, events, event_id, tmin=-2, tmax=3, baseline=(-2, 0), preload=True)
            epochs = epochs.crop(tmin=0, tmax=3)
        elif baseline_correction and extraction_section == False:
            epochs = mne.Epochs(raw, events, event_id, tmin=-2, tmax=2, baseline=(-2, 0), preload=True)
            epochs = epochs.crop(tmin=-1, tmax=2)
        elif baseline_correction == False and extraction_section:
            epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=3, baseline=None, preload=True)
        else:
            epochs = mne.Epochs(raw, events, event_id, tmin=-1, tmax=2, baseline=None, preload=True)

        epoch_data = epochs.get_data()
        epoch_data = epoch_data[:, :, :samplefreq*cut_out_time] # 時間ステップを理論値に変換
        # サンプル数が合わない被験者を除外(除外しているが運動が変わった時のダブルチェックの意味で)
        if epoch_data.shape[0] != 15:
            pass
        else:
            if type_of_movement == "fist":
                labels = np.array([1 if event[-1] == event_id['right_fist'] else 0 for event in epochs.events])
                subject_epoch_data.append(epoch_data)
                subject_labels.append(labels)
            else:
                labels = np.array([2 if event[-1] == event_id['both_fist'] else 3 for event in epochs.events])
                subject_epoch_data.append(epoch_data)
                subject_labels.append(labels)

    if subject_epoch_data:
        subject_epoch_data = np.concatenate(subject_epoch_data,axis=0)
        subject_labels = np.concatenate(subject_labels,axis=0)
        # データ型を定義
        dt = np.dtype([("epoch_data", np.float64, subject_epoch_data.shape[1:]), ("label", np.int64)])
        # 構造化配列を作成
        structured_data = np.empty(len(subject_epoch_data), dtype=dt)
        structured_data["epoch_data"] = subject_epoch_data
        structured_data["label"] = subject_labels

        save_path = tmp_save_path / f"S{num+1:03d}"  # 保存先のディレクトリ名を連番に調整
        os.makedirs(save_path,exist_ok=True)
        np.save(f"{save_path}/{type_of_movement}_movement.npy",structured_data)
        num += 1
    else:
        pass
