import glob
import os
import pickle

import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TFHUB_CACHE_DIR'] = 'C:\debug'
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
import tensorflow_text as text
from official.nlp import optimization
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from transformers import AutoTokenizer, AutoModel
import numpy as np


def encode_data(x_data):

    #ALBERT CODE

    # print("Downloading...")
    # preprocessor = hub.KerasLayer(
    #     "http://tfhub.dev/tensorflow/albert_en_preprocess/2")
    # encoder = hub.KerasLayer(
    #     "https://tfhub.dev/tensorflow/albert_en_xlarge/2",
    #     trainable=False)
    # print("Done!")
    # text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    # encoder_inputs = preprocessor(text_input)
    # outputs = encoder(encoder_inputs)
    # net = outputs['pooled_output']
    # bert = tf.keras.Model(text_input, net)
    # bert.summary()

    # print("encoding data...")
    # encoded_data = bert.predict(x_data, batch_size=1, verbose=1)
    # encoded_data = bert(x_data).numpy()
    # print("Done!")

    # HUGGINGFACE Bio_ClinicalBERT

    print("encoding data...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    embeddings_inputs = tokenizer(x_data.tolist())["input_ids"]
    embeddings_inputs = [torch.tensor(x).reshape((1, -1)) for x in embeddings_inputs]

    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    embedded_data = [model.forward(x)['pooler_output'].detach().numpy() for x in embeddings_inputs]
    print("Done!")

    return np.array(embedded_data).reshape(-1, 768)


def build_classifier_model():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(x_data.shape[-1])))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(256, activation='gelu'))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(128, activation='gelu'))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(64, activation='gelu'))
    model.add(tf.keras.layers.Dropout(.2))

    model.add(tf.keras.layers.Dense(8, activation='softmax'))
    return model




if __name__ == '__main__':
    model_name = 'model7'
    model_path = os.path.join("models", model_name)
    epochs = 1000
    batch_size = 32

    # x_data, y_data = format_data()
    # idxs = np.where(y_data[y_data.any() == 1])[1]
    #
    # x_data, y_data = x_data[idxs], y_data[idxs]
    # x_data = encode_data(x_data)
    #
    # np.save("xdata.npy", x_data)
    # np.save("ydata.npy", y_data)

    x_data, y_data = np.load("xdata.npy"), np.load("ydata.npy")
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = [tf.metrics.CategoricalAccuracy(), tfa.metrics.MatthewsCorrelationCoefficient(8), tfa.metrics.F1Score(8)]

    optimizer = optimization.create_optimizer(init_lr=3e-5,
                                              num_train_steps=(len(X_train)/batch_size)*epochs,
                                              num_warmup_steps=((len(X_train)/batch_size)*epochs)*.1,
                                              optimizer_type='adamw')

    classifier_model = build_classifier_model()
    # classifier_model = tf.keras.models.load_model(model_path, compile=False)
    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    classifier_model.summary()

    # print(classifier_model.evaluate(X_train, y_train))
    # print(classifier_model.evaluate(X_test, y_test))

    def scheduler(epoch):
        if 0.001 * np.exp(0.01 * -epoch) > 0.000005:
            return 0.001 * np.exp(0.01 * -epoch)
        return 0.000005


    func = lambda e: max(0.00005 * np.exp(0.01 * -e), .000001)
    func2 = lambda e: .000001
    callbacks = [LearningRateScheduler(func, verbose=1)]

    # ModelCheckpoint(filepath=model_path, monitor='loss', save_best_only=True)
    history = classifier_model.fit(x=np.array(X_train), y=np.array(y_train), epochs=epochs,
                                   batch_size=batch_size, validation_data=(X_test, y_test))

    classifier_model.save(model_path)

    def results(idx):
        l = classifier_model(tf.constant([x_data[idx]])).numpy()[0]
        print([f'{i:2.0%}:{j:2.0%}' for i, j in zip(l, y_data[idx])])

    for i in range(0, 50):
        results(i)


    with open("history.pkl", 'wb') as file:
        pickle.dump(history, file)

