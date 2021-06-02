import os
import pickle
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

x_data, y_data = np.load("xdata.npy"), np.load("ydata.npy")
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

distribution = {}
one_hot_dict = {0: 'TrIP', 1: 'TrWP', 2: 'TrCP', 3: 'TrAP', 4: 'TrNAP', 5: 'PIP', 6: 'TeRP', 7: 'TeCP'}
for i in range(len(y_train)):
    relation = one_hot_dict[np.argmax(y_train[i])]
    if relation not in distribution:
        distribution[relation] = 0
    distribution[relation] += 1

plt.figure(1)
plt.bar(np.arange(8), distribution.values())
plt.xticks(np.arange(8), list(distribution.keys()))
plt.title("Training distribution")


distribution = {}
one_hot_dict = {0: 'TrIP', 1: 'TrWP', 2: 'TrCP', 3: 'TrAP', 4: 'TrNAP', 5: 'PIP', 6: 'TeRP', 7: 'TeCP'}
for i in range(len(y_test)):
    relation = one_hot_dict[np.argmax(y_test[i])]
    if relation not in distribution:
        distribution[relation] = 0
    distribution[relation] += 1

plt.figure(2)
plt.bar(np.arange(8), distribution.values())
plt.xticks(np.arange(8), list(distribution.keys()))
plt.title("Validation distribution")
plt.show()


history = pickle.load(open("history.pkl", 'rb'))

trn_loss = history['loss']
val_loss = history['val_loss']
nepochs = len(trn_loss)

# summarize history for loss
plt.figure()
plt.plot(np.arange(nepochs), gaussian_filter1d(trn_loss, sigma=2), linestyle='dashed', label='Train')
plt.plot(np.arange(nepochs), gaussian_filter1d(val_loss, sigma=2), linewidth=2.0, label='Validation')

plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.show()

trn_f1 = np.array(history['f1_score'])
val_f1 = np.array(history['val_f1_score'])

plt.figure()
color = plt.cm.Set2(np.linspace(0, 1, 8))
labels = ['TrIP', 'TrWP', 'TrCP', 'TrAP', 'TrNAP', 'PIP', 'TeRP', 'TeCP']
for i in range(8):
    plt.plot(np.arange(nepochs), gaussian_filter1d(trn_f1[:, i], sigma=8), c=color[i], linestyle='dashed')
    plt.plot(np.arange(nepochs), gaussian_filter1d(val_f1[:, i], sigma=10), c=color[i], label=labels[i])

plt.ylabel('F1_Score')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.show()

trn_mcc = np.array(history['MatthewsCorrelationCoefficient'])
val_mcc = np.array(history['val_MatthewsCorrelationCoefficient'])

plt.figure()
color = plt.cm.Set2(np.linspace(0, 1, 8))
labels = ['TrIP', 'TrWP', 'TrCP', 'TrAP', 'TrNAP', 'PIP', 'TeRP', 'TeCP']
for i in range(8):
    plt.plot(np.arange(nepochs), trn_mcc[:, i], c=color[i], linestyle='dashed')
    plt.plot(np.arange(nepochs), val_mcc[:, i], c=color[i], label=labels[i])

plt.ylabel('Matthews Correlation Coefficient')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.show()
