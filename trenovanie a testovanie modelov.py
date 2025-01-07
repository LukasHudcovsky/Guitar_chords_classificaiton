import numpy as np
from numpy import mean, absolute
import os
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score


#inicializacia poctu atributov pre jednotilve datasety
num_svm = 5
num_rf = 15
num_lr = 40
data_svm = np.zeros((502, num_svm + 1))
data_rf = np.zeros((502, num_rf + 1))
data_nn = np.zeros((502, num_rf + 1))
data_lr = np.zeros((502, num_lr + 1))
j = 0

#prechadzanie cez vsetky nahravky a vytvaranie datasetov pre jednotlive modely
for i in range (0, 2):
    if i==0:
        path = "archive/chords_dataset/major"
    else:
        path = "archive/chords_dataset/minor"
    for file in os.listdir(path):
        if file.endswith(".wav"):
            full_path = path + "/" + file

            #nacitanie suboru
            sr, wav_data = wavfile.read(full_path)

            #prevedenie na mono (jeden kanal)
            wav_data = wav_data[:, 0]

            #Fourierova transformacia signalu na ziskanie pola frekvencii
            N = len(wav_data)
            yfreq = np.abs(rfft(wav_data))
            xfreq = rfftfreq(N, 1 / sr)

            #hladanie extremov na ziskanie najvyznamnejsich frekvencii
            limit = np.max(yfreq)*0.1
            extremes, _ = find_peaks(yfreq, height=limit, distance=20)

            #nechavame rozny pocet najvyznamnejsich frekvencii pre kazdy model
            frequencies = xfreq[extremes]
            if len(frequencies)>num_svm:
                length1 = num_svm
            else:
                length1 = len(frequencies)
            if len(frequencies)>num_rf:
                length2 = num_rf
            else:
                length2 = len(frequencies)
            if len(frequencies)>num_lr:
                length3 = num_lr
            else:
                length3 = len(frequencies)

            data_svm[j, :length1] = frequencies[:length1]
            data_rf[j, :length2] = frequencies[:length2]
            data_lr[j, :length3] = frequencies[:length3]

            #hodnota cielovej funkcie
            if i==0:
                data_svm[j, -1] = 1
                data_lr[j, -1] = 1
                data_rf[j, -1] = 1
            j += 1

data_nn = data_rf

#premesanie dat a rozdelenie na trenovaciu a testovaciu cast
np.random.shuffle(data_lr)
np.random.shuffle(data_svm)
np.random.shuffle(data_rf)
np.random.shuffle(data_nn)

x_train_svm = data_svm[:400, :-1]
y_train_svm = data_svm[:400, -1]
x_train_rf = data_rf[:400, :-1]
y_train_rf = data_rf[:400, -1]
x_train_lr = data_lr[:400, :-1]
y_train_lr = data_lr[:400, -1]
x_train_nn = data_nn[:400, :-1]
y_train_nn = data_nn[:400, -1]

x_test_svm = data_svm[400:, :-1]
y_test_svm = data_svm[400:, -1]
x_test_rf = data_rf[400:, :-1]
y_test_rf = data_rf[400:, -1]
x_test_lr = data_lr[400:, :-1]
y_test_lr = data_lr[400:, -1]
x_test_nn = data_nn[400:, :-1]
y_test_nn = data_nn[400:, -1]

#normalizacia dat pre neuronovu siet
mean1 = np.mean(x_train_nn, axis=0)
std1 = np.std(x_train_nn, axis=0)
mean2 = np.mean(x_test_nn, axis=0)
std2 = np.std(x_test_nn, axis=0)
x_train_nn = (x_train_nn - mean1) / std1
x_test_nn = (x_test_nn - mean2) / std2

#vytvaranie a trenovanie modelu neuronovej siete
mlp = Sequential()
mlp.add(Dense(10, activation='relu', input_dim=x_train_nn.shape[1]))
mlp.add(Dense(8, activation='relu'))
mlp.add(Dropout(0.1))
mlp.add(Dense(5, activation='relu'))
mlp.add(Dense(2, activation='softmax'))

mlp.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001),  metrics=['accuracy'])

history = mlp.fit(x_train_nn, keras.utils.to_categorical(y_train_nn), epochs=30, validation_split=0.1, verbose=True)

train_score = mlp.evaluate(x_train_nn, keras.utils.to_categorical(y_train_nn))
test_score = mlp.evaluate(x_test_nn, keras.utils.to_categorical(y_test_nn))

#vytvaranie a trenovanie ostatnych modelov
cv = KFold(n_splits=5, shuffle=True)

svc = svm.SVC(C=20, kernel='rbf', gamma=0.0002)
score1 = cross_val_score(svc, x_train_svm, y_train_svm, cv=cv)
svc.fit(x_train_svm, y_train_svm)

rfc = RandomForestClassifier(n_estimators=30, max_depth=15)
score2 = cross_val_score(rfc, x_train_rf, y_train_rf, cv=cv)
rfc.fit(x_train_rf, y_train_rf)

lr = LogisticRegression(random_state=0)
score3 = cross_val_score(lr, x_train_lr, y_train_lr, cv=cv)
lr.fit(x_train_lr, y_train_lr)

#vypisanie vysledkov pre jednotlive modely
print("Cross validation score for SVM: {}".format(mean(absolute(score1))))
print("Cross validation score for Random Forest: {}".format(mean(absolute(score2))))
print("Cross validation score for Logistic Regression: {}".format(mean(absolute(score3))))
print("Training accuracy for Neural Network: {}\n".format(train_score[1]))

print("Testing accuracy for SVM: {}".format(svc.score(x_test_svm, y_test_svm)))
print("Testing accuracy for Random Forest: {}".format(rfc.score(x_test_rf, y_test_rf)))
print("Testing accuracy for Logistic Regression: {}".format(lr.score(x_test_lr, y_test_lr)))
print("Testing accuracy for Neural Network: {}".format(test_score[1]))