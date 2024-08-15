import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Sound_Class:
    classes = ['PD', 'HC']

    def __init__(self, data_name, sample_rate, trim_silence=True, silence_duration=1):
        self.data_name = data_name
        self.sample_rate = sample_rate
        self.trim_silence = trim_silence
        self.silence_duration = silence_duration

    def load_data(self):
        sr = self.sample_rate
        samples_rates = []
        data = []
        labels = []

        for c in self.classes:
            path = os.path.join(self.data_name, c)

            for voice in os.listdir(path):
                voice_path = os.path.join(path, voice)
                voice_data, samplerate = librosa.load(voice_path, dtype=np.float32, sr=sr)
                voice_data = (voice_data * 32767).astype(np.int16)

                if self.trim_silence:
                    silence_samples = int(self.silence_duration * samplerate)
                    silence_threshold = np.percentile(np.abs(voice_data[:silence_samples]), 90)
                    start_idx = next((idx for idx, val in enumerate(voice_data) if abs(val) > silence_threshold), None)
                    if start_idx is not None:
                        voice_data = voice_data[start_idx:]

                samples_rates.append(samplerate)
                data.append(voice_data)

                if c == 'PD':
                    labels.append(1)
                else:
                    labels.append(0)

        df = pd.DataFrame({"Samples Rates": samples_rates, "voices": data, "labels": labels})
        return df

    def train_test_split_func(self, X, Y, test_size, random_state):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state,
                                                            stratify=Y)
        return X_train, X_test, y_train, y_test

    def split_to_x_y(self, data):
        X = data.drop('labels', axis=1)
        Y = data['labels']
        return X, Y

    def MFCC(self, audio, sample_rate):
        mfccs_features = librosa.feature.mfcc(y=audio.astype(float), sr=sample_rate, n_mfcc=30)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features

    def get_feature(self, X):
        features_mfcc = []
        for signal, sr in zip(X['voices'], X['Samples Rates']):
            features_mfcc.append(self.MFCC(signal, sr))
        return features_mfcc

    def classifier(self, classifier_name, object_class, train_features, y_train, test_features, y_test, n_splits=4):
        object_class.fit(train_features, y_train)
        predictions_test = object_class.predict(test_features)
        accuracy_test = accuracy_score(y_test, predictions_test)
        print("Accuracy on full of data " + classifier_name + ' : ', accuracy_test)

    def ConfusionMatrix(self, model, X_test, Y_test):
        predictions = model.predict(X_test)
        cm = confusion_matrix(Y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()