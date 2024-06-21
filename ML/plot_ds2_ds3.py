import numpy as np
import matplotlib.pyplot as plt

ds2_path = "C:/Users/keisokuken_kaisekiPC/Documents/Python_project/ML_project/ML/ref_data/DWT_data/ds_2/S032/DWT_data.npy"
ds3_path = "C:/Users/keisokuken_kaisekiPC/Documents/Python_project/ML_project/ML/ref_data/DWT_data/ds_3/S032/DWT_data.npy"

ds2_data = np.load(ds2_path, allow_pickle=True).item()["left_right_fist"]["epoch_data"]
ds3_data = np.load(ds3_path, allow_pickle=True).item()["left_right_fist"]["epoch_data"]

for i in range(30):
    fig, ax = plt.subplots(2, 1, figsize=(10,20))
    ax[0].plot(ds2_data[0].T[0], marker="o", linestyle="-")
    ax[1].plot(ds3_data[0].T[0], marker="o", linestyle="-")
    plt.show()