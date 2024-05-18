from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
from scipy.signal import resample, butter, filtfilt
from module1.preprocessing import *

def path_name(type_of_movement):
    if type_of_movement == "left_right_fist":
        return "fist_movement.npy"
    elif type_of_movement == "fists_feet":
        return "both_fists_feet_movement.npy"
    else:
        raise ValueError(f"Invalid type_of_movement: {type_of_movement}. Expected 'fist' or 'feet'.")

def select_electrode(number_of_ch=64):
    all_ch = ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6','Cp5', 'Cp3', 'Cp1',
            'Cpz', 'Cp2', 'Cp4', 'Cp6', 'Fp1', 'Fpz', 'Fp2', 'Af7', 'Af3', 'Afz', 'Af4','Af8', 'F7', 'F5', 'F3', 'F1','Fz',
            'F2', 'F4','F6', 'F8', 'Ft7', 'Ft8', 'T7', 'T8','T9', 'T10', 'Tp7', 'Tp8', 'P7', 'P5', 'P3', 'P1','Pz','P2',
            'P4','P6', 'P8', 'Po7','Po3', 'Poz', 'Po4', 'Po8', 'O1', 'Oz', 'O2', 'Iz']
    extracted_ch = []
    ch_idx = np.arange(0, number_of_ch).tolist()
    for i in ch_idx:
        extracted_ch.append(all_ch[i])
    return ch_idx, extracted_ch

def distribute_labels(ch_idx, all_epoch_data, all_labels, n_class, number_of_ch=int):
    data_list = []
    label_list = []

    for label_val in range(n_class):
        indices = np.where(all_labels == label_val)[0]
        if number_of_ch == 64:
            data = all_epoch_data[indices]
        else:
            data = all_epoch_data[indices][:, ch_idx, :]
        labels = all_labels[indices]
        data_list.append(data)
        label_list.append(labels)

    # when rest is included(not now using)
    # rest_hand_indices = np.where(all_labels == 3)[0]
    # if number_of_ch == 64:
    #     rest_hand_data = all_epoch_data[rest_hand_indices]
    # else:
    #     rest_hand_data = all_epoch_data[rest_hand_indices][:, ch_idx, :]
    # rest_hand_labels = all_labels[rest_hand_indices]

    # data_list.append(rest_hand_data)
    # label_list.append(rest_hand_labels)

    # Combine data
    combined_data = np.concatenate(data_list, axis=0)
    combined_labels = np.concatenate(label_list, axis=0)

    return combined_data, combined_labels

def dwt(sig, level, t, ch_name, wavelet, plot = bool):
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    nyq_freq = len(coeffs[level])
    sub_band_freq = {}
    if plot == True:
        fig, ax = plt.subplots(level+2, 1, figsize=(12, 8))
        # Plot the original signal
        ax[0].plot(t, sig)
        ax[0].set_title(f'Original Signal for {ch_name}')
        for i in range(level+1):
            if i == 0:
                sub_band_freq[f"A{level-i}"] = f"{0} - {nyq_freq / 2**(level+1)} Hz"
                # Plot the approximation part
                ax[i+1].plot(coeffs[i])
                ax[i+1].set_title(f"A{level}")
            else:
                detail_num = level+1-i
                sub_band_freq[f"D{detail_num}"] = f"{nyq_freq / 2**(detail_num + 1)} - {nyq_freq / 2**(detail_num)} Hz"
                # Plot the detail part
                ax[i+1].plot(coeffs[i])
                ax[i+1].set_title(f'D{level+1-i}')
        print(sub_band_freq)
        plt.tight_layout()
        plt.show()
    else:
        pass
    return coeffs

def Preprocessing(preprocessing_type):
    if preprocessing_type == "d":
        preprocessing_dir = "DWT_data"
    elif preprocessing_type == "e":
        preprocessing_dir = "Envelope_data"
    elif preprocessing_type == "b" :
        preprocessing_dir = "BPF_data"
    return preprocessing_dir

def execute_dwt(epoch_data, decompose_level, d_num):
    details_per_epoch = []
    all_details = []
    for ch in range(epoch_data.shape[0]):
        # Decomposition level
        level = decompose_level
        # Get DWT results
        coeffs = dwt(epoch_data[ch, :], level, t, extracted_ch[ch], "db4", plot = False)
        details = []
        # Combine any specific detail coefficients
        for n in range(1, d_num+1, 1):
            details.extend(coeffs[n])
        details_per_epoch.append(details) # Creates an element for each channel per sample
    all_details.append(details_per_epoch) # Creates elements for the number of batches
    return all_details

def execute_envelope(epoch_data, ds, samplerate=160):
    all_envelope = []
    filterd_data = filter(epoch_data, ds, samplerate) # BPF
    envelope_data = extract_envelope(filterd_data, samplerate, 1)
    all_envelope.append(envelope_data)
    return all_envelope

def execute_bpf(epoch_data, ds):
    all_filtered_data = []
    filtered_data = filter(epoch_data, ds, samplerate) # BPF
    all_filtered_data.append(filtered_data)  # Add band-pass filtered data to the list
    return all_filtered_data

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
samplerate = 160 # Sampling frequency
wave_list = ["mu", "beta", "mu_beta"]
downsampling_levels = [2, 3] # Downsampling levels

extraction_section = True # True if the extraction section does not include rest, False if it does
baseline_correction = True # Whether to perform baseline correction
ext_sec = "move_only" if extraction_section else "rest_move"
baseline = "baseline_true" if baseline_correction else "baseline_false"

n_class = 4 # Number of classes for classification to be set later, so here it's the maximum value of 4
type_of_movement_1 = "left_right_fist"
type_of_movement_2 = "fists_feet"

current_dir = Path.cwd() # Get the current directory
eeg_data_dir = current_dir / "ML" / "ref_data" / "ML_data"

preprocessing_type= "e" # d(DWT), e(Envelope), b(BPF)

d_num = 3 # Number of details to be obtained (from top down {D4,D3...})
decompose_level = 5 # Decomposition level

# Get data path
subject_dirs = []
for i in range(104):
    subject_dirs.extend(eeg_data_dir.glob(f"S{i+1:03}"))

ch_idx, extracted_ch = select_electrode()

file_name_1 = path_name(type_of_movement_1)
file_name_2 = path_name(type_of_movement_2)

for subject_dir in subject_dirs:
    all_data_list = []
    subject_id = subject_dir.name # Subject name

    # Combine paths and load data
    data_1 = np.load(subject_dir / file_name_1)
    data_2 = np.load(subject_dir / file_name_2)

    for ds in downsampling_levels:
        preprocessing_dir = Preprocessing(preprocessing_type)
        all_data_dict = {}
        for data, movement_type in zip([data_1, data_2], [type_of_movement_1, type_of_movement_2]):
            epoch_data = np.stack(data["epoch_data"])
            labels = data["label"]
            X, y = distribute_labels(ch_idx, epoch_data, labels, n_class, number_of_ch=64)
            for i, epoch_data in enumerate(X):
                # Apply anti-aliasing filter
                cutoff = samplerate // (2 * ds)  # Set cutoff frequency
                filtered_epoch_data = apply_antialiasing_filter(epoch_data, cutoff, samplerate)
                epoch_data = resample(filtered_epoch_data, filtered_epoch_data.shape[1] // ds, axis=1) # Downsample the data
                t = np.linspace(0, epoch_data.shape[1]/(samplerate/ds), epoch_data.shape[1])

                if preprocessing_dir == "DWT_data":
                    all_data, dir_name = execute_dwt(epoch_data, decompose_level, d_num)
                elif preprocessing_dir == "Envelope_data":
                    all_data = execute_envelope(epoch_data, ds)
                elif preprocessing_dir == "BPF_data":
                    all_data = execute_bpf(epoch_data, ds)
                all_data_list.extend(all_data)
            all_preprocessing_data = np.array(all_data_list).transpose(0,2,1) # Reshape to (number of trials, number of sample points, number of channels)
            all_data_list = []

            # Add data and labels for each movement state to the dictionary
            all_data_dict[movement_type] = {"epoch_data": all_preprocessing_data, "labels": y}

        if preprocessing_dir == "DWT_data":
            save_dir = current_dir / "ML" / "ref_data" / "DWT_data" / f"ds_{ds}" / subject_id
        elif preprocessing_dir == "Envelope_data":
            save_dir = current_dir / "ML" / "ref_data" / "Envelope_data" / f"ds_{ds}" / subject_id
        elif preprocessing_dir == "BPF_data":
            save_dir = current_dir / "ML" / "ref_data" / "BPF_data" / f"ds_{ds}" / subject_id
        os.makedirs(save_dir, exist_ok=True)
        output_file = save_dir / f"{preprocessing_dir}.npy"
        np.save(output_file, all_data_dict)
