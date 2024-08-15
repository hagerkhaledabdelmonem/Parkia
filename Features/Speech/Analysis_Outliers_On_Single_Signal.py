import numpy as np
from scipy import stats
import Functions


def detect_outliers(signal, threshold):
    z_scores = np.abs(stats.zscore(signal))
    outliers = np.where(z_scores > threshold)[0]
    return outliers


def remove_outliers(signal, outliers):
    # Convert signal to numpy array
    signal_array = np.array(signal)
    # Remove outliers using numpy's delete() function
    cleaned_signal = np.delete(signal_array, outliers)

    print("Original data size:", len(signal_array))
    print("Cleaned data size:", len(cleaned_signal))
    print('*' * 50)


df = Functions.load_data("silenceRemoved", 8000)

threshold = 10  # Threshold for z-score
for i, signal in enumerate(df['voices']):
    outliers = detect_outliers(signal, threshold)
    remove_outliers(signal, outliers)
