# import the needed libraries
from sys import getsizeof
import tensorflow as tf
import requests
from base64 import b64encode
import numpy as np
import time
from scipy.io import wavfile
import scipy.signal as signal
import os
import pandas as pd
import json
from datetime import datetime

# download and unzip the dataset from the web
zip_path = tf.keras.utils.get_file(origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                                   fname='mini_speech_commands.zip',
                                   extract=True,
                                   cache_dir='.', cache_subdir='data')

# get the list of selected tracks from the given textual file
test_files=pd.read_csv('./kws_test_split.txt', sep="\n")
test_files=np.array(test_files.values)
test_files=test_files.squeeze()

# get the labels from textual file
with open('./labels.txt') as f:
    LABELS = np.array(eval(f.read()))

# declare the method to read the wav files, plus the related label index
def read(file_path,LABELS):

        parts = tf.strings.split(file_path, '/')
        label = parts[-2]
        label_id=np.argwhere(LABELS==label)[0][0]
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio,label_id

# declare constant values
sampling_rate = 8000
frame_length = 256
frame_step = 128
num_mel_bins = 16
lower_frequency = 20
upper_frequency = 4000
num_coefficients = 10
num_spectrogram_bins = (frame_length) // 2 + 1

# compute once the transformation matrix
linear_to_mel_weight_matrix = tf.cast(tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins, num_spectrogram_bins, sampling_rate,
                lower_frequency, upper_frequency),tf.float32)

# istantiate once the model
interpreter = tf.lite.Interpreter(model_path="./kws_dscnn_True.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# declare method to preprocess and mae predictions
def preprocess_infer(audio):

# repeat constant variables
    sampling_rate = 8000
    frame_length = 256
    frame_step = 128
    num_mel_bins = 16
    lower_frequency = 20
    upper_frequency = 4000
    num_coefficients = 10
    num_spectrogram_bins = (frame_length) // 2 + 1

# make visible internally the prevoiusly calculated variables
    global interpreter
    global linear_to_mel_weight_matrix
    global input_details
    global output_details

# resample signal, apply stft, obtain spectrograms
    audio = signal.resample_poly(audio, sampling_rate, 16000)
    stft = tf.signal.stft(tf.cast(audio, tf.dtypes.float32), frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)

# obtain log mel spectrograms, then resize and cut to the wanted MFCCs
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    log_mel_spectrogram=tf.expand_dims(log_mel_spectrogram,-1)
    log_mel_spectrogram=tf.image.resize(log_mel_spectrogram,[49,16])
    c=tf.squeeze(log_mel_spectrogram)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(c)
    mfccs = mfccs[..., :num_coefficients]
    mfccs = tf.expand_dims(mfccs, -1)

#create the dataset (single sample) to be submitted to the net
    X = tf.data.Dataset.from_tensor_slices([[mfccs]])

# input the sample to the interpreter, and get the normalized (softmax) confidence scores
    for i in X:
        interpreter.set_tensor(input_details['index'],i)
        interpreter.invoke()
        my_output = interpreter.get_tensor(output_details['index'])
    return np.exp(my_output)/np.sum(np.exp(my_output))

# declare the confidence threshold
tr=0.5

# initialize arrays that store performance rates
ground_truth=[]
predictions=[]
times=[]
bytes_sent = []

# declare the URL for communication
url = 'http://0.0.0.0:8080/prediction'

# sequentially analye samples
for aux in test_files:
    start=time.time()
    x,y=read(aux,LABELS)
    ground_truth.append(y)
    pred_stats=preprocess_infer(x)
    end=time.time()
    times.append(end-start)

# check if we need to send to the server
    if np.max(pred_stats)<tr:

# encode the audio, and send it into SenML+JSON format
        x=tf.audio.encode_wav([x], 16000, name=None)
        audio_string = b64encode(x.numpy())
        now =datetime.now()
        audio_string =audio_string.decode()
        body = {"bn": f"{url}","bt": str(now.timestamp()),"e": [{"n": "record", "vd": audio_string}]}
        body=json.dumps(body)
        bytes_sent.append(getsizeof(body))

        r = requests.post(url,body)

# check if the communication went successfully, and in gthat case store the received prediction
        if r.status_code == 200:
            body = r.json()
            pred=np.argwhere(LABELS==body["pred"])
# else, print the error on screen
        else:
            print('Error:', r.status_code)

# in case the sending is not needed, we take the local prediction
    else:
        pred=np.argmax(pred_stats)

    predictions.append(pred)

# compute and print the overall performances
predictions = np.array(predictions)
ground_truth = np.array(ground_truth)
mask = predictions==ground_truth
print("Accuracy:","{:.3%}".format(sum(mask)/len(mask)))
print("Time:","{:.2f}".format(np.mean(times)*1000),"ms")
print("Communication Cost:","{:.3f}".format( np.sum(bytes_sent)/1024**2), "MB")





