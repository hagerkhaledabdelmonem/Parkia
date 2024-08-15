import io
import os
import warnings
from io import BytesIO
from flask import Flask, request
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import struct
import librosa
import joblib
from pydub import AudioSegment
from pydub.silence import split_on_silence

warnings.filterwarnings('ignore')
app = Flask(__name__)

sample_rate = 8000
silence_duration = 1

# Feature1 Functions (HandDrawing)
def png_to_jpg(png_data):
    # convert png to jpeg
    jpg_output = BytesIO()
    Image.fromarray(png_data).save(jpg_output, "JPEG")
    jpg_data = jpg_output.getvalue()
    jpg_output.close()
    image_pil = Image.open(io.BytesIO(jpg_data))
    image_array = np.array(image_pil)
    return image_array

def preprocess_image(images):
    image_list = []
    for img in images:
        # Convert the image data to a numpy array
        image_np = np.frombuffer(img, dtype=np.uint8)
        # Decode the image
        decoded_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        jpg_image = png_to_jpg(decoded_image)
        img = cv2.cvtColor(jpg_image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (180, 180))
        img = img / 255.0
        image_list.append(img)
    return image_list

def hand_drawing_prediction(images):
    preprocessed_images = preprocess_image(images)
    image_fusion = np.stack((preprocessed_images[0], preprocessed_images[1], preprocessed_images[2]), axis=-1)
    image_fusion = np.expand_dims(image_fusion, axis=-1)  # Add a channel dimension
    image_fusion = np.expand_dims(image_fusion, axis=0)
    Model = load_model("saved models\HandDrawing\CNN_fusion.h5")
    output_data = Model.predict(image_fusion)
    return output_data

# Feature2 Functions (Speech)
def parse_wav_bytes(byte_data):
    header_size = 44
    if len(byte_data) <= header_size:
        raise ValueError('Invalid WAV file')

    pcm_data = byte_data[header_size:]

    values = []
    for i in range(0,len(pcm_data),2):
        value = struct.unpack_from('<h', pcm_data, i)[0]
        values.append(value)
    return np.array(values)

def save_wav_file(byte_data, file_path):
    with open(file_path, 'wb') as f:
        f.write(byte_data)

def remove_silence_librosa(byte_data, top_db=28):
    parsed = parse_wav_bytes(byte_data)
    intervals = librosa.effects.split(parsed, top_db=top_db)
    non_silent_audio = np.concatenate([parsed[start:end] for start, end in intervals])
    return non_silent_audio

def remove_silence(audio_path, silence_threshold = -50):
    audio = AudioSegment.from_wav(audio_path)
    # Split the audio into segments
    segments = split_on_silence(audio, silence_thresh=silence_threshold)
    # Combine the non-silent segments into a new audio file
    output = AudioSegment.silent()
    for segment in segments:
        output += segment
    # Export the result to a new WAV file
    output_path = "afterRemoveSilence.wav"
    output.export(output_path, format="wav")
def audio_prediction(audio):
    # Step 1: Removing Silence
    audio_samples = remove_silence_librosa(audio)
    silence_samples = int(silence_duration * sample_rate)
    # Calculate the energy threshold for silence detection
    silence_threshold = np.percentile(np.abs(audio_samples[:silence_samples]), 90)
    # Find the index of the first sample that exceeds the silence threshold
    start_idx = next((idx for idx, val in enumerate(audio_samples) if abs(val) > silence_threshold), None)
    if start_idx is not None:
        # Trim audio to remove silence at the beginning
        audio_samples = audio_samples[start_idx:]

    # Step 3: Feature Extraction
    mfccs_features2 = librosa.feature.mfcc(y=audio_samples.astype(float), sr=sample_rate,
                                           n_mfcc=40)  # Convert back to float
    mfccs_scaled_features2 = np.mean(mfccs_features2.T, axis=0)
    # Step 4: Load the saved model
    loaded_pipeline = joblib.load("saved models\Speech\model.joblib")
    y_pred = loaded_pipeline.predict(mfccs_scaled_features2.reshape(1, -1))

    return y_pred

@app.route('/appTest', methods=['POST'])
def test():
    # Receive the data from the app
    image_data = [request.files['image1'].read(), request.files['image2'].read(), request.files['image3'].read()]
    audio_data = request.files['audio'].read()

    # Hand drawing prediction

    shapes_output = hand_drawing_prediction(image_data)
    shapes_prediction = ""
    if shapes_output[0][0] < shapes_output[0][1]:
        shapes_prediction = "Patient"
    else:
        shapes_prediction = "Healthy"
    print("Output (Hand Drawing): ", shapes_prediction)

    # Audio Prediction
    audio_output = audio_prediction(audio_data)
    audio_result = ""
    if audio_output == 1:
        audio_result = 'Patient'
    elif audio_output == 0:
        audio_result = 'Healthy'
    print("Output (Audio): ", audio_result)
    app_output = [shapes_prediction, audio_result]
    return app_output


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
