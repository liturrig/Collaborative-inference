#import section
import cherrypy
import tensorflow as tf
import requests
from base64 import b64decode
import numpy as np
from scipy.io import wavfile
import scipy.signal as signal
import os
import pandas as pd
import json


#predictor class
class Predictor(object):
    exposed = True
    #costants
    sampling_rate = 16000
    frame_length = 640
    frame_step = 320
    num_mel_bins = 40
    lower_frequency = 20
    upper_frequency = 4000
    num_coefficients = 10
    num_spectrogram_bins = (frame_length) // 2 + 1
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    num_mel_bins, num_spectrogram_bins, sampling_rate,
                    lower_frequency, upper_frequency)
    interpreter = tf.lite.Interpreter(model_path="./kws_dscnn_True.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]


    #read label file
    with open('./labels.txt') as f:
        LABELS = np.array(eval(f.read()))


    #preprocessing and inferencing 
    def preprocess_infer(self, audio):

        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])
        #creating sftf
        stft = tf.signal.stft(audio, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)
        mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        #creating mfccs
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]
        mfccs = tf.expand_dims(mfccs, -1)
        #transforming mfccs to tensor
        X = tf.data.Dataset.from_tensor_slices([[mfccs]])
        #inference
        for i in X:

            self.interpreter.set_tensor(self.input_details['index'],i)
            self.interpreter.invoke()
            my_output = self.interpreter.get_tensor(self.output_details['index'])
        #return the prediction
        return np.argmax(my_output)
    #POST method
    def POST(self, *path, **query):
        
        #read body
        body = cherrypy.request.body.read()
        audio=json.loads(body)
        #extract the audio
        audio_encoded=audio["e"][0]["vd"]
        #decoding
        record =b64decode(audio_encoded)
        record=tf.audio.decode_wav(record)
        #preprocess and inference step
        prediction = self.preprocess_infer(record[0][0])
        #transform into label
        word=self.LABELS[prediction]
        #json transformation
        body = json.dumps({"pred": word})
        return body

    def PUT(self, *path, **query):
        pass

    def GET(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass
#main
if __name__== '__main__':
    conf = {'/':{'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(Predictor(), '', conf)
    cherrypy.config.update({'server.socket_host':'0.0.0.0'})
    cherrypy.config.update({'server.socket_port':8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
    

