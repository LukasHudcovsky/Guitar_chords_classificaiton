import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import os
from tensorflow import keras
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from numpy import mean, absolute


num = 5
j = 0
data = np.zeros((502, num + 1))

#prechadzanie cez vsetky nahravky a vytvaranie datasetu
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

            #nechavame len prvych num najvyznamnejsich frekvencii, ostatne zahadzujeme
            frequencies = xfreq[extremes]
            if len(frequencies)>num:
                length = num
            else:
                length = len(frequencies)

            data[j, :length] = frequencies[:length]

            #hodnota cielovej funkcie
            if i==0:
                data[j, -1] = 1
            j += 1

#premesanie dat a rozdelenie na trenovaciu a testovaciu cast
np.random.shuffle(data)

x_train = data[:400, :-1]
y_train = data[:400, -1]

x_test = data[400:, :-1]
y_test = data[400:, -1]

cv = KFold(n_splits=5, shuffle=True)

#vyber najlepsich hyperparametrov pre SVM
best = 0
g_best = 0.0001
c_best = 1
for i in range(1, 40):
    g = 1
    for j in range(0, 15):
        g = g*2
        svc = svm.SVC(C=i, kernel='rbf', gamma=0.0001*g)
        score = cross_val_score(svc, x_train, y_train, cv=cv)
        if mean(absolute(score)) > best:
            best = mean(absolute(score))
            g_best = 0.0001*g
            c_best = i
            print("{}, {}".format(c_best, g_best))

svc = svm.SVC(C=c_best, kernel='rbf', gamma=g_best)
score = cross_val_score(svc, x_train, y_train, cv=cv)
print("Mean absolute score for SVM: {}".format(mean(absolute(score))))
print(c_best)
print(g_best)


#vyber najlepsich hyperparametrov pre Random Forest
best = 0
n_best = 10
d_best = 5
for i in range(10, 40):
    for j in range(5, 20):
        rfc = RandomForestClassifier(n_estimators=i, max_depth=j)
        score = cross_val_score(rfc, x_train, y_train, cv=cv)
        if mean(absolute(score)) > best:
            best = mean(absolute(score))
            n_best = i
            d_best = j
            print("{}, {}".format(n_best, d_best))

rfc = RandomForestClassifier(n_estimators=n_best, max_depth=d_best)
score = cross_val_score(rfc, x_train, y_train, cv=cv)
print("Mean absolute score for Random Forest: {}".format(mean(absolute(score))))
print(n_best)
print(d_best)


#pre kazdy model (SVM, Random forest, Logistic regression) vyberame pocet najvyznamnejsich frekvencii (atributov dat) tak aby mal co najvacsiu uspesnost
num = 0
best1 = 0
best2 = 0
best3 = 0
best_num1 = 0
best_num2 = 0
best_num3 = 0
values = np.zeros((10, 3))

#iterovanie cez pocty frekvencii
for k in range(0, 10):
    num += 5
    j = 0
    data = np.zeros((502, num + 1))

    for i in range (0, 2):
        if i==0:
            path = "archive/chords_dataset/major"
        else:
            path = "archive/chords_dataset/minor"
        for file in os.listdir(path):
            if file.endswith(".wav"):
                full_path = path + "/" + file

                sr, wav_data = wavfile.read(full_path)
                wav_data = wav_data[:, 0]

                N = len(wav_data)
                yfreq = np.abs(rfft(wav_data))
                xfreq = rfftfreq(N, 1 / sr)

                limit = np.max(yfreq)*0.1
                extremes, _ = find_peaks(yfreq, height=limit, distance=20)

                frequencies = xfreq[extremes]
                if len(frequencies)>num:
                    length = num
                else:
                    length = len(frequencies)

                data[j, :length] = frequencies[:length]
                if i==0:
                    data[j, num] = 1
                j += 1

    np.random.shuffle(data)

    x_train = data[:400, :-1]
    y_train = data[:400, -1]

    #vypocet skore pre jednotlive modely
    svc = svm.SVC(C=20, kernel='rbf', gamma=0.0002)
    score1 = cross_val_score(svc, x_train, y_train, cv=cv)
    rfc = RandomForestClassifier(n_estimators=30, max_depth=15)
    score2 = cross_val_score(rfc, x_train, y_train, cv=cv)
    lr = LogisticRegression(random_state=0)
    score3 = cross_val_score(lr, x_train, y_train, cv=cv)

    values[k, 0] = mean(absolute(score1))
    values[k, 1] = mean(absolute(score2))
    values[k, 2] = mean(absolute(score3))

    if mean(absolute(score1)) > best1:
        best1 = mean(absolute(score1))
        best_num1 = num
    if mean(absolute(score2)) > best2:
        best2 = mean(absolute(score2))
        best_num2 = num
    if mean(absolute(score3)) > best3:
        best3 = mean(absolute(score3))
        best_num3 = num

for i in range(0, 10):
    print("SVM: {} {}".format((i+1)*5, values[i, 0]))
    print("RandomForestClassifier: {} {}".format((i+1)*5, values[i, 1]))
    print("LogisticRegression: {} {}".format((i+1)*5, values[i, 2]))
    print()

print("\n")
print("LogisticRegression: {} {}".format(best_num1, best1))
print("SVM: {} {}".format(best_num2, best2))
print("RandomForestClassifier: {} {}".format(best_num3, best3))

#tabulka uspesnosti modelov pre jednotlive pocty vstupnych frekvencii
freqs=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
models=["SVM", "Random forest", "Logistic regresion"]

table = pd.DataFrame(values, columns=models, index=freqs)
print(table)

#graf uspesnosti modelov pre jednotlive pocty vstupnych frekvencii
plt.plot(freqs, values[:, 0], color='blue')
plt.plot(freqs, values[:, 1], color='lime')
plt.plot(freqs, values[:, 2], color='red')
plt.xlabel('Number of used frequencies')
plt.ylabel('Cross validation score')
plt.legend(['SVM', 'Random forest', 'Logistic regresion'])
plt.show()


#vyberame vhodny pocet atributov dat pre Neuronovu siet
num = 0
results = np.zeros((10))

for k in range(0, 10):
    num += 5
    j = 0
    data = np.zeros((502, num + 1))

    for i in range (0, 2):
        if i==0:
            path = "archive/chords_dataset/major"
        else:
            path = "archive/chords_dataset/minor"
        for file in os.listdir(path):
            if file.endswith(".wav"):
                full_path = path + "/" + file

                sr, wav_data = wavfile.read(full_path)
                wav_data = wav_data[:, 0]

                N = len(wav_data)
                yfreq = np.abs(rfft(wav_data))
                xfreq = rfftfreq(N, 1 / sr)

                limit = np.max(yfreq)*0.1
                extremes, _ = find_peaks(yfreq, height=limit, distance=20)

                frequencies = xfreq[extremes]
                if len(frequencies)>num:
                    length = num
                else:
                    length = len(frequencies)

                data[j, :length] = frequencies[:length]
                if i==0:
                    data[j, num] = 1
                j += 1

    np.random.shuffle(data)

    x_train = data[:400, :-1]
    y_train = data[:400, -1]

    x_test = data[400:, :-1]
    y_test = data[400:, -1]

    #normalizacia dat
    mean1 = np.mean(x_train, axis=0)
    std1 = np.std(x_train, axis=0)
    mean2 = np.mean(x_test, axis=0)
    std2 = np.std(x_test, axis=0)

    x_train = (x_train - mean1) / std1
    x_test = (x_test - mean2) / std2

    #vytvaranie modelu
    mlp = Sequential()
    mlp.add(Dense(10, activation='relu', input_dim=x_train.shape[1]))
    mlp.add(Dense(8, activation='relu'))
    mlp.add(Dropout(0.1))
    mlp.add(Dense(5, activation='relu'))
    mlp.add(Dense(2, activation='softmax'))

    mlp.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001),  metrics=['accuracy'])

    history = mlp.fit(x_train, keras.utils.to_categorical(y_train), epochs=30, validation_split=0.1, verbose=True)

    train_score = mlp.evaluate(x_train, keras.utils.to_categorical(y_train))
    results[k] = train_score[1]

#uspesnost neuronovej siete pre jednotlive pocty atributov
print("Neural network:")
for i in range(0, 10):
    print("{} frequencies:   train acc: {}".format((i+1)*5, results[i]))