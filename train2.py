import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from format_data import format_data
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding, Dropout, GlobalAveragePooling1D, Dense, LSTM
import tensorflow_addons as tfa
import numpy as np
from sklearn.model_selection import train_test_split


def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

    model = tf.keras.Sequential()
    model.add(text_input)
    model.add(vectorize_layer)
    model.add(Embedding(max_features + 1, embedding_dim))
    model.add(Dropout(0.2))
    model.add(LSTM(16))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8))
    return model


if __name__ == '__main__':
    model_name = 'model.h5'
    model_path = os.path.join("models", model_name)
    epochs = 50
    bsize = 32

    # x_data, y_data = format_data()
    # np.save("xdata.npy", x_data)
    # np.save("ydata.npy", y_data)
    x_data, y_data = np.load("xdata.npy"), np.load("ydata.npy")
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    max_features = 1600
    embedding_dim = 128
    sequence_length = 100

    vectorize_layer = TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)
    vectorize_layer.adapt(x_data)

    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.metrics.CategoricalAccuracy(), tfa.metrics.MatthewsCorrelationCoefficient(8), tfa.metrics.F1Score(8)]

    classifier_model = build_classifier_model()
    classifier_model.summary()
    classifier_model.compile(optimizer='adam', loss=loss, metrics=metrics)
    classifier_model.fit(x=np.array(X_train), y=y_train, epochs=epochs)

    def results(idx):
        l = classifier_model(tf.constant([x_data[idx]])).numpy()[0]
        print([f'{i:2.0%}' for i in l])

    results(0)
    results(5)
    results(12)
    results(44)
    results(115)

