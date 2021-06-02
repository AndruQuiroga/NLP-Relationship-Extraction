import os
import tensorflow as tf
from tensorflow.python.keras.models import load_model

loss = "mse"
opt = "adam"


def build_model(filepath):
    if os.path.exists(filepath):
        if input(f"{filepath} exists! Are you sure you want to overwrite? [Y]/n ").lower() in ['yes', 'y']:
            _build_model(filepath)
    else:
        _build_model(filepath)


def _build_model(filepath):
    # model = Sequential()
    #
    # activation = 'relu'
    #
    # # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(nbins, nbins, 1)))
    # # model.add(MaxPooling2D())
    # # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    # # model.add(MaxPooling2D())
    # # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    # # model.add(MaxPooling2D())
    # # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    # # model.add(MaxPooling2D())
    # # model.add(Flatten())
    #
    # # model.add(MaxPooling2D(input_shape=(nbins, nbins, 1)))
    # model.add(Flatten(input_shape=(nbins, nbins, 1)))
    # model.add(Dense(units=128, activation=activation))
    # for i in range(6):
    #     model.add(ComplexDense(units=nbins, activation=activation))
    #
    # model.add(ComplexDense(units=dims))
    #
    # model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    # model.summary()
    # model.save(filepath)
    pass


def model_load(model_path, multi_GPU=False):
    if not multi_GPU:
        model = load_model(model_path)
        model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        return model

    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of GPUs: {strategy.num_replicas_in_sync}')
    with strategy.scope():
        model = load_model(model_path)
        model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        return model
