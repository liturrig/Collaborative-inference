# import the needed libraries
from DoSomething import DoSomething
import time
import json

# declare the subscriber class
class Subscriber(DoSomething):

# declare the wanted behavior on the receival of new messages
    def notify(self, topic, msg):

# load the message and extract the dictionary and timestamp related to SenML+JSON format
        input_json = json.loads(msg)
        time = input_json["bt"]
        e=eval(input_json["e"])
# format the message to be printed on screen
        if e[0]["n"]=="Temperature":
            print("("+time.split(".")[0]+")",e[0]["n"],"Alert: Predicted="+"{:.1f}".format(e[0]["v_pred"])+"°"+e[0]["u"][0],"Actual="+"{:.1f}".format(e[1]["v_real"])+"°"+e[0]["u"][0])
        else:
            print("("+time.split(".")[0]+")",e[0]["n"],"Alert: Predicted="+"{:.1f}".format(e[0]["v_pred"])+e[0]["u"][0],"Actual="+"{:.1f}".format(e[1]["v_real"])+e[0]["u"][0])


# istantiate a new subscriber node, and subscribe it to the 'alert' channel
test = Subscriber('subscriber 1')
test.run()
test.myMqttClient.mySubscribe("/290464/alert")
while True:
    time.sleep(1)
