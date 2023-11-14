from tensorflow import keras
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Input, AveragePooling1D, Activation
from keras.models import Model
from keras.layers import SeparableConv1D, BatchNormalization
import requests
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from skopt.space import Real, Categorical, Integer
from keras.wrappers.scikit_learn import KerasClassifier
from skopt import BayesSearchCV
from keras.callbacks import EarlyStopping

def one_dim_CNN_model(input_shape, n_class, stride=2, conv1_filters=32, kernel1_size=20, conv2_filters=32, kernel2_size=9, conv3_filters=32,
                    kernel3_size=5, dense1_nodes=26, dense2_nodes=13, dropout_rate=0.2, optimizer='adam', learning_rate=0.001):
    if optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError("Unknown optimizer")

    input_layer = Input(shape=input_shape)
    x = Conv1D(28, 20, strides=2, activation='relu', padding='same', name="L1")(input_layer)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = SeparableConv1D(20, 9, strides=2, activation='relu', padding='valid', name="L2")(x)
    x = AveragePooling1D(2, name="L3")(x)
    x = Conv1D(28, 5, strides=1, activation='relu', padding='same', name="L4")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = SeparableConv1D(28, 3, strides=1, activation='relu', padding='valid', name="L5")(x)
    x = Flatten(name="L6")(x)
    x = Dense(20, activation='relu', name="L7")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(12, activation='relu', name="L8")(x)
    output_layer = Dense(n_class, activation='softmax', name="L9")(x)

    model = Model(input_layer, output_layer)
    if n_class == 2:
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    return model

def create_model_builder(input_shape, n_class):
    def model_builder(input_shape, n_class, conv1_filters=64, kernel1_size=20, stride1=2, stride2=1, conv2_filters=32, kernel2_size=15, conv3_filters=64,
                    kernel3_size=10, conv4_filters=32, kernel4_size=20, dense1_nodes=26, dense2_nodes=13, dropout_rate=0.2, optimizer='adam', learning_rate=0.001):
        return one_dim_CNN_model(input_shape, n_class, conv1_filters, kernel1_size, stride1, stride2, conv2_filters, kernel2_size, conv3_filters,
                    kernel3_size, conv4_filters, kernel4_size, dense1_nodes, dense2_nodes, dropout_rate, optimizer, learning_rate)
    return model_builder

def Bayesian_Opt(X, y, n_class, n_iter=150, cv=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = to_categorical(y_train, num_classes=n_class)
    y_test = to_categorical(y_test, num_classes=n_class)

    timesteps = X_train.shape[1]
    input_dim = X_train.shape[2]

    input_shape = (timesteps, input_dim)

    model_builder = create_model_builder(input_shape, n_class)
    # モデルをラップ
    one_dim_CNN_model_wrap = KerasClassifier(build_fn=model_builder, verbose=0)

    # ハイパーパラメータの範囲を指定
    classification_param_dict = {'conv1_filters': Integer(8, 64),
            'kernel1_size': Integer(3,20),
            'conv2_filters': Integer(16, 64),
            'kernel2_size': Integer(3,20),
            'conv3_filters': Integer(16, 64),
            'kernel3_size': Integer(3,20),
            'conv4_filters': Integer(16, 64),
            'kernel4_size': Integer(3,20),
            'dense1_nodes': Integer(16, 32),
            'dense2_nodes': Integer(8, 32),
            'stride1': Integer(1, 5),
            'stride2': Integer(1, 5),
            'dropout_rate': Real(0.2, 0.5),
            'optimizer': Categorical(['adam'])
            }
    callbacks_list = [EarlyStopping(monitor="val_loss", patience=4)]
    # ベイズ最適化を実施
    Bayes_search_CNN = BayesSearchCV(estimator=one_dim_CNN_model_wrap, search_spaces=classification_param_dict, n_iter=n_iter, n_jobs=3, cv=cv, verbose=2)
    Bayes_search_CNN.fit(X_train, y_train, batch_size=12, epochs=200, callback=callbacks_list)

    # ベイズ最適化の結果を保存
    best_params = Bayes_search_CNN.best_params_
    # 最適なパラメータでモデルを再トレーニング
    optimized_model = model_builder(**best_params)
    optimized_model.fit(X_train, y_train, batch_size=12, epochs=200, verbose=0, callbacks=callbacks_list)  # epochsは適切な値に設定してください

    # テストデータで評価
    evaluation = optimized_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {evaluation[0]}, Test accuracy: {evaluation[1]}")
    return best_params, evaluation

#PythonからLINEへ通知を送る関数
def line_notify(message):
    line_notify_token = 'pzvwGSaMkHyipTsMij8xIYnNmRVDuoufvkHJbOzVvHx'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    payload = {'message': message} #引数として自由に入力可能
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    requests.post(line_notify_api, data=payload, headers=headers)
