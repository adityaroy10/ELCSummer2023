import pyrebase
import pandas as pd
import collections
from datetime import datetime,timedelta
import requests,time,random,json,redis,websocket

config = {
  'apiKey': "AIzaSyAMzyV5TrnaxZxhONkEzb1UArSlTpxR-II",
  'authDomain': "capstonetest-2a851.firebaseapp.com",
  'projectId': "capstonetest-2a851",
  'databaseURL':"https://capstonetest-2a851-default-rtdb.firebaseio.com/",
  'storageBucket': "http://capstonetest-2a851.appspot.com/",
  'messagingSenderId': "31150681068",
  'appId': "1:31150681068:web:6bc219e86d7e6bdfaf6ec9",
  'measurementId': "G-71ETB9N43N"
};
firebase = pyrebase.initialize_app(config)
database = firebase.database()

ws = websocket.WebSocket()
ws.connect(r'ws://localhost:8000/ws/polData/')
while(True):
    time.sleep(5)
    a = database.get()
    b = list(a.val().items())[0][1]
    val3 = {}
    val3['value'] = int(list(b.values())[-1].split(',')[0])
    val3['value2'] = float(list(b.values())[-1].split(',')[3].split('\\')[0])
    val3['value3'] = int(list(b.values())[-1].split(',')[1])
    val3['value4'] = float(list(b.values())[-1].split(',')[2])
    f = open("alert.txt", "a")
    if val3['value']<60:
      f.write(f"Blood Pressure has decreased! Reading:{val3['value']}\n")
    if val3['value']>80:
      f.write(f"Blood Pressure has increased! Reading:{val3['value']}\n")
    if val3['value2']<35:
      f.write(f"Temperature has decreased! Reading:{val3['value2']}\n")
    if val3['value2']>39:
      f.write(f"Temperature has increaed! Reading:{val3['value2']}\n")
    if val3['value3']<90:
      f.write(f"Oxygen Saturation has decreased! Reading:{val3['value3']}\n")
    if val3['value4']<600 and val3['value4']>400:
      f.write(f"ECG is in critical condition! Reading:{val3['value4']}\n")
    f.close()
    ws.send(json.dumps(val3))
