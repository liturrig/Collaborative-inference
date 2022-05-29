# import the needed libraries
import tensorflow as tf
import requests
from base64 import b64encode
import time
import argparse

## FIRST SERVICE

# read the first model from local storage
model_folder="./models/dht_mlp.tflite"
with open(model_folder, 'rb') as f:
    modello=f.read()

# encode the model to be transmitted
model_encoded = b64encode(modello)
model_string = model_encoded.decode()

# declare the URL, with the given path
url = 'http://localhost:8080/add'

# compose the JSON dict and send the message
body={"model": model_string, "name": model_folder.split("/")[2]}
r = requests.post(url,body)

# if the message has arrived, load the response and print it on screen
if r.status_code == 200:
    #body = r.json()
    print("completed storage")

# else, print the error message
else:
    print('Error:', r.status_code)

# read the second model from local storage, encrypt it and send it to the server (exactly as we did previously)
model_folder="./models/dht_cnn.tflite"
with open(model_folder, 'rb') as f:
    modello=f.read()

model_encoded = b64encode(modello)
model_string = model_encoded.decode()

url = 'http://localhost:8080/add'

body={"model": model_string, "name": model_folder.split("/")[2]}
r = requests.post(url,body)

if r.status_code == 200:
    #body = r.json()
    print("completed storage")
else:
    print('Error:', r.status_code)

## SECOND SERVICE

# declare the URL, with the given path, and send the request
url = 'http://localhost:8080/list'
r = requests.get(url)

# if the message has arrived, load the response and print it on screen
if r.status_code == 200:
    body = r.json()
    print("The list of models is: " + ", ".join(body["models"]))

# check that there are exacly two models
    if len(body["models"])!=2:
        raise Exception("There are not two models!")

# else, print the error message
else:
    print('Error:', r.status_code)

# THIRD SERVICE

# select the wanted model to make the inference
model="dht_mlp.tflite"

# declare the wanted thresholds
temp=0.1
hum=0.2

# compose the wanted URL (giving path + query), then send the request
url = 'http://localhost:8080/predict?model='+model+"&tthres="+str(temp)+"&hthres="+str(hum)
r = requests.get(url)

# if the message has arrived, load the response and print it on screen
if r.status_code == 200:
    print('Successful operation')

# else, print the error message
else:
    print('Error:', r.status_code)

