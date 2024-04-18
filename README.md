## EEG Motor Imagery classification with one-dimensional-convolutional neural network <img src="https://img.shields.io/badge/-Python-F9DC3E.svg?logo=python&style=flat">
This project facilitates EEG classification using Python and Keras.
It allows for an immersive experience in EEG signal processing and machine learning for motor imagery classification.

### How to use
Firstly, You need to clone this repositry. 

#### Data formatting
1. Choose the type of movement EEG data you wish to work with (right fist movement, right fist movement, both fists movement and both feet movement) by setting the "type_of_movement" variable.
2. Once executed, the formatted EEG data will be available in (ML/ref_data/ML_data/).

#### Signal Preprocessing Phase
1. Open the project folder and select "save_data.py".
2. Modify the "preprocessing_type" variable to choose your preferred method of preprocessing.
3. After execution, a processed signal folder will be created in "ref_data" (which denotes reference data).

#### Machine Learning Phase
1. Open the project folder and select "main.py".
2. Change the "preprocessing_type" variable to select your preferred method of preprocessing.
3. The "ds" variable allows you to choose the downsampling rate.
