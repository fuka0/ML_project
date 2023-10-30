import numpy as np
from scipy.signal import butter,filtfilt,hilbert

def filter(data, ds, sample_rate, order=6):
    if ds == 2:
        lowcut = 8
        highcut = 30
        nyquist = 0.5 * sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
    elif ds == 3:
        lowcut = 8
        nyquist = 0.5 * sample_rate
        low = lowcut / nyquist
        b, a = butter(order, low, btype='high')

    if data.ndim >= 2:
        filtered_data = np.empty_like(data)
        for ch_idx in range(data.shape[0]):
            filtered_data[ch_idx] = filtfilt(b, a, data[ch_idx], axis=0)
    else:
        filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def extract_envelope(data,samplerate,low,order=2):
    envelope_data = np.abs(hilbert(data, axis=1))
    nyquist = 0.5 * samplerate
    low = low / nyquist
    b, a = butter(order, low, btype='low')
    for ch_idx in range(envelope_data.shape[0]):
        envelope_data[ch_idx] = filtfilt(b, a, envelope_data[ch_idx], axis=0)
    return envelope_data

# バターワースフィルタでのアンチエイリアシングフィルタの設定
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# アンチエイリアシングフィルタを適用する関数
def apply_antialiasing_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y