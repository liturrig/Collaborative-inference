
#import section
from DoSomething import DoSomething
import time
from datetime import datetime
from MyMQTT import MyMQTT
import numpy as np
import adafruit_dht
import time
import RPi.GPIO as GPIO
from board import D4
import tensorflow as tf
import os
import cherrypy
import json
from base64 import b64decode


#store model class
class PostTheModel(object):
    exposed = True

    def POST(self, *path, **query):
        #check path and query
        if len(path) != 0 or len(query)==0:

            raise cherrypy.HTTPError(400, 'Wrong query')

        #extract the model
        body = query["model"]

        record = b64decode(body)

        #write the model to file
        with open(f"./models/{query['name']}", 'wb') as fp:
            fp.write(record)

        pass

    def GET(self, *path, **query):
        pass
    def PUT(self, *path, **query):
        pass
    def DELETE(self, *path, **query):
        pass

#list stored models class
class ListStoredModels(object):
    exposed = True

    def GET(self, *path, **query):
        #check of the path
        if len(path) != 0:
            raise cherrypy.HTTPError(400, 'Wrong query')
        if len(query) != 0:
            raise cherrypy.HTTPError(400, 'Wrong query')
        #list models in model folder
        folder=os.listdir("./models/")
        output={"models": folder}
        output_str = json.dumps(output)
        return output_str

    def POST(self, *path, **query):
        pass
    def PUT(self, *path, **query):
        pass
    def DELETE(self, *path, **query):
        pass
# predict measures class
class PredictAndSendAlert(object):
    exposed = True
    #dht parameters
    dht_sensor_port = 4                     # Connect the DHT sensor to port 4
    device = "pi-003"                       # Host name of the data collector d$
    GPIO.setmode(GPIO.BCM)                  # Use the Broadcom pin numbering
    GPIO.setup(dht_sensor_port, GPIO.IN)    # DHT sensor port as input
    dht_device = adafruit_dht.DHT11(D4)
    MEAN = np.array([9.107597, 75.904076], dtype=np.float32) #mean of dataset
    STD = np.array([ 8.654227, 16.557089], dtype=np.float32) #standard deviation of dataset

    def GET(self, *path, **query):
        #check of the path
        if len(path) != 0:
            raise cherrypy.HTTPError(400, 'Wrong query')
        if len(query) != 3:
            raise cherrypy.HTTPError(400, 'Wrong query')

        #set variables obtained by query
        model = query.get('model')
        tempt = query.get('tthres')
        humt = query.get('hthres')


        #check variables
        if model is None:
            raise cherrypy.HTTPError(400, 'Wrong query')
        if tempt is None:
            raise cherrypy.HTTPError(400, 'Wrong query')
        if humt is None:
            raise cherrypy.HTTPError(400, 'Wrong query')
        tempt = float(tempt) #casting
        humt  = float(humt ) #casting


        #instance and run the object
        test = DoSomething("publisher 1")
        test.run()


        #tflite model
        interpreter = tf.lite.Interpreter(model_path=f"./models/{model}")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        #set variables
        recordings=[]
        my_output=None


        while True:

            #record temperature and humidity by sensor
            temp=self.dht_device.temperature
            hum=self.dht_device.humidity
            #append temp and hum to list
            recordings.append((np.array([float(temp),float(hum)])-self.MEAN)/self.STD)
            #check if number of recordings are 6 or more
            if len(recordings)>6:
                #remove the first couple of recordings
                recordings.pop(0)
            if len(recordings)==6:
                if my_output is not None:
                    #calulate the error
                    maes=np.abs([temp,hum]-my_output)
                    #if the error over take the threshold
                    #temperature case
                    if maes[0][0]>tempt:
                        posixTime_sens=datetime.now()
                        t1=(posixTime_sens-posixTime_pred).seconds
                        #SenML JSON format
                        output={"bn":"http:/0.0.0.0","bt":str(posixTime_pred),"e":str([{"n":  "Temperature",  "u":  "Cel",  "t": "0","v_pred":my_output[0][0]},{"n":  "Temperature",  "u":  "Cel",  "t": f"{t1}",  "v_real": temp}])}
                        output = json.dumps(output)
                        #publishing
                        test.myMqttClient.myPublish("/290464/alert", output)
                    #humidity case
                    if maes[0][1]> humt:
                        posixTime_sens=datetime.now()
                        t2=(posixTime_sens-posixTime_pred).seconds
                        #SenML JSON format
                        output={"bn":"http:/0.0.0.0","bt":str(posixTime_pred),"e":str([{"n":  "Humidity",  "u":  "%RH",  "t": f"0","v_pred":my_output[0][1]},{"n":  "Humidity",  "u":  "%RH",  "t": f"{t2}",  "v_real": hum}])}
                        output = json.dumps(output)
                        #publishing
                        test.myMqttClient.myPublish("/290464/alert", output)
                #from numpy to tensor
                X= tf.data.Dataset.from_tensor_slices(np.array([[recordings]],dtype=np.float32))
                #inference
                for i in X:
                    interpreter.set_tensor(input_details['index'],i)
                    interpreter.invoke()
                    my_output = interpreter.get_tensor(output_details['index'])
                    posixTime_pred=datetime.now()

            #wait 1 second
            time.sleep(1)

        pass

    def POST(self, *path, **query):
        pass
    def PUT(self, *path, **query):
        pass
    def DELETE(self, *path, **query):
        pass


#main
if __name__== '__main__':
    conf = {'/':{'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(PostTheModel(), '/add', conf)
    cherrypy.tree.mount(ListStoredModels(), '/list', conf)
    cherrypy.tree.mount(PredictAndSendAlert(), '/predict', conf)
    cherrypy.config.update({'server.socket_host':'0.0.0.0'})
    cherrypy.config.update({'server.socket_port':8080})
    cherrypy.engine.start()
    cherrypy.engine.block()






