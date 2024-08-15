import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import librosa
import librosa.display
import numpy as np
import pandas as pd


def load_data(data_name, sr, threshold, trim_silence=True, silence_duration=1, detect_outliers=True):
    classes = ['PD', 'HC']
    samples_rates = []
    data = []
    labels = []
    i = 1
    for class_label in classes:
        path = os.path.join(data_name, class_label)
        for voice_file in os.listdir(path):
            voice_path = os.path.join(path, voice_file)

            # Load audio data
            try:
                voice_data, samplerate = librosa.load(voice_path, dtype=np.float32, sr=sr)
                voice_data = (voice_data * 32767).astype(np.int16)  # Convert to 16-bit integer
            except Exception as e:
                print(f"Error loading voice file {voice_file}: {e}")
                continue

            # Calculate number of samples before any processing
            num_samples_before = len(voice_data)

            if trim_silence:
                # Calculate number of samples corresponding to the silence duration
                silence_samples = int(silence_duration * samplerate)
                # Calculate the energy threshold for silence detection
                silence_threshold = np.percentile(np.abs(voice_data[:silence_samples]), 90)
                # Find the index of the first sample that exceeds the silence threshold
                start_idx = next((idx for idx, val in enumerate(voice_data) if abs(val) > silence_threshold), None)
                if start_idx is not None:
                    # Trim audio to remove silence at the beginning
                    voice_data = voice_data[start_idx:]
                else:
                    print(f"Warning: No start index found for voice {voice_file}")

            # Calculate number of samples after trimming silence
            num_samples_after_silence = len(voice_data)

            if len(voice_data) > 0:
                data.append(voice_data)
                samples_rates.append(samplerate)
                if class_label == 'PD':
                    labels.append(1)
                else:
                    labels.append(0)

                # Create a DataFrame for the current voice
                voice_df = pd.DataFrame({"Samples Rates": [samplerate], "voices": [voice_data], "labels": [labels[-1]]})

                if detect_outliers:
                    # Detect outliers based on Z-score for each voice record
                    z_scores = np.abs(
                        (voice_df['voices'].iloc[0] - voice_df['voices'].iloc[0].mean()) / voice_df['voices'].iloc[
                            0].std())
                    outlier_mask = (z_scores > threshold)  # Adjust Z-score threshold as needed

                    # Filter out rows identified as outliers
                    voice_df = voice_df.loc[~outlier_mask[:len(voice_df['voices'])]]

                # Calculate number of samples after removing outliers
                num_samples_after_outliers = len(voice_data)
                voice_df = pd.DataFrame({"Samples Rates": [samplerate], "voices": [voice_data], "labels": [labels[-1]]})

                # Print information about the current voice file
                print(str(i) + ') Voice:')
                print(f"  Samples before trimming silence: {num_samples_before}")
                print(f"  Samples after trimming silence: {num_samples_after_silence}")
                print(f"  Samples after removing outliers: {num_samples_after_outliers}")
                print("--------------------------------------------")
                i += 1
            else:
                print(f"Warning: Skipping empty voice {voice_file}")

    return data


threshold = 10  # Threshold for z-score
loaded_df = load_data("silenceRemoved", 8000, threshold)
