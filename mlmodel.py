import pandas as pd
import numpy as np
import random,math
import warnings
import itertools
from datetime import datetime,timedelta,date
import pyrebase
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import requests,time,random,json,redis,websocket,collections
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
warnings.filterwarnings("ignore")
from prophet import Prophet


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

config1 = {
  'apiKey': "AIzaSyAGLHV0lJ0qhPmcoVO3HulVPwfRf0OpJxo",
  'authDomain': "capstonefuture.firebaseapp.com",
  'databaseURL': "https://capstonefuture-default-rtdb.firebaseio.com",
  'projectId': "capstonefuture",
  'storageBucket': "capstonefuture.appspot.com",
  'messagingSenderId': "609555345653",
  'appId': "1:609555345653:web:9049fb485f8846474715bc",
  'measurementId': "G-0VBHKNTPCT"
};

firebase = pyrebase.initialize_app(config)
database = firebase.database()

firebase1 = pyrebase.initialize_app(config1)
database1 = firebase1.database()

# ws = websocket.WebSocket()
# ws.connect(r'ws://localhost:8000/ws/predData/')

for i in range(10):
  # time.sleep(10)
  a = database.get()
  b = list(a.val().items())[0][1]
  data = list(b.values())

  value = []
  value2 = []
  value3 = []
  value4 = []

  for i in data:
    value.append(int(i.split(',')[0]))
    value2.append(float(i.split(',')[3].split('\\')[0]))
    value3.append(int(i.split(',')[1]))
    value4.append(float(i.split(',')[2]))

  df = pd.DataFrame(columns=['Time','Blood Pressure','Temperature','Oxygen Saturation','Blood Glucose'])
  df['Time'] = b.keys()
  df2 = pd.DataFrame(a.val().values())
  df['Blood Pressure'] = value
  df['Temperature'] = value2
  df['Oxygen Saturation'] = value3
  df['Blood Glucose'] = value4
  df['Time'] = pd.to_datetime(df['Time'])
  print(df)

  #ARIMA
  #Blood Pressure
  stepwise_fit_BP = auto_arima(df['Blood Pressure'], trace=True,suppress_warnings=True)

  train_count = math.floor((max(df.count()))*0.7)
  test_count = int((max(df.count()))-train_count)
  train = df[:train_count+1]
  test = df[train_count+1:]

  model_arima_BP=ARIMA(train['Blood Pressure'],order=stepwise_fit_BP.get_params().get("order"))
  model_arima_BP=model_arima_BP.fit()

  start=len(train)
  end=len(train)+len(test)-1
  pred=model_arima_BP.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')

  rmse_arima_BP = np.mean((pred - test['Blood Pressure'])**2)**.5

  #LSTM
  #Blood Pressure
  def sampling(sequence, n_steps):
      X,Y = list(), list()
      for i in range(len(sequence)):
          sam = i + n_steps
          if sam > len(sequence)-1:
              break
          x, y = sequence[i:sam],sequence[sam]
          X.append(x)
          Y.append(y)
      return np.array(X), np.array(Y)
  n_steps = 3
  X, Y = sampling(df['Blood Pressure'].tolist(), n_steps)
  X_train, Y_train = sampling(df['Blood Pressure'][:train_count+1].tolist(), n_steps)
  X_test, Y_test = sampling(df['Blood Pressure'][train_count+1:].tolist(), n_steps)

  model_LSTM_BP = Sequential()
  model_LSTM_BP.add(LSTM(60, activation='relu', input_shape=(3,1)))
  model_LSTM_BP.add(Dense(1))
  model_LSTM_BP.compile(optimizer='adam', loss='mse')

  X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
  model_LSTM_BP.fit(X_train, Y_train, epochs=200, verbose=0)

  X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

  ypred = model_LSTM_BP.predict(X_test, verbose=0)

  rmse_lstm_BP = np.mean((ypred - Y_test)**2)**.5

  look_back = 15
  close_data = X.reshape((-1))
  def predic(num_prediction, model):
      prediction_list = close_data[-look_back:]

      for _ in range(num_prediction):
          x = prediction_list[-look_back:]
          x = x.reshape((1, look_back, 1))
          out = model_LSTM_BP.predict(x)[0][0]
          prediction_list = np.append(prediction_list, out)
      prediction_list = prediction_list[look_back-1:]
      return prediction_list

  #Prophet
  #Blood Pressure
  proph_BP = pd.DataFrame(columns=['ds','y'])
  proph_BP['ds'] = df['Time']
  proph_BP['y'] = df['Blood Pressure']

  model_prophet_BP = Prophet(interval_width=0.95)
  model_prophet_BP.fit(proph_BP)
  future_dates = model_prophet_BP.make_future_dataframe(periods=36, freq='MS')
  forecast = model_prophet_BP.predict(future_dates)

  ans = datetime.strptime(datetime.today().strftime("%Y-%m-%d %H:%M:%S"),"%Y-%m-%d %H:%M:%S")
  final_BP = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]['ds']
  rmse_prophet_BP = np.mean((forecast.loc[:, 'yhat'] - proph_BP['y'])**2)**.5
  try:
    ans1 = int(predic(1,model_LSTM_BP).tolist()[0])
  except:
    ans1=0
  try:
    ans2 = int(model_arima_BP.forecast().tolist()[0])
  except:
    ans2=0
  try:
    ans3 = int(model_prophet_BP.predict(pd.DataFrame({'ds':[str(ans)]}))['yhat'].tolist()[0])//120
  except:
    ans3=0
  ls = [ans1,ans2,ans3]

  if rmse_lstm_BP<rmse_arima_BP and rmse_lstm_BP<rmse_prophet_BP:
      ind = 0

  if rmse_arima_BP<rmse_lstm_BP and rmse_arima_BP<rmse_prophet_BP:
      ind = 1

  if rmse_prophet_BP<rmse_arima_BP and rmse_prophet_BP<rmse_lstm_BP:
      ind = 2

  if (ls[ind]!=0 or ls[ind]!=0.00) and (ls[ind]>50 or ls[ind]>50.00):
    bp = ls[ind]
  else:
    ls.pop(ind)
    if ls[0]!=0 or ls[0]!=0.00:
      bp = ls[0]
    else:
      bp = ls[1]

  #ARIMA
  #Temperature
  stepwise_fit_Temp = auto_arima(df['Temperature'], trace=True,suppress_warnings=True)

  model_arima_Temp=ARIMA(train['Temperature'],order=stepwise_fit_Temp.get_params().get("order"))
  model_arima_Temp=model_arima_Temp.fit()

  start=len(train)
  end=len(train)+len(test)-1

  pred=model_arima_Temp.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')

  rmse_arima_Temp = np.mean((pred - test['Temperature'])**2)**.5

  #LSTM
  #Temperature
  X, Y = sampling(df['Temperature'].tolist(), n_steps)
  X_train, Y_train = sampling(df['Temperature'][:train_count+1].tolist(), n_steps)
  X_test, Y_test = sampling(df['Temperature'][train_count+1:].tolist(), n_steps)

  model_LSTM_Temp = Sequential()
  model_LSTM_Temp.add(LSTM(60, activation='relu', input_shape=(3,1)))
  model_LSTM_Temp.add(Dense(1))
  model_LSTM_Temp.compile(optimizer='adam', loss='mse')

  X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

  model_LSTM_Temp.fit(X_train, Y_train, epochs=200, verbose=0)

  X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

  ypred = model_LSTM_Temp.predict(X_test, verbose=0)

  rmse_lstm_Temp = np.mean((ypred - Y_test)**2)**.5

  close_data = X.reshape((-1))

  #Prophet
  #Temperature
  proph_Temp = pd.DataFrame(columns=['ds','y'])
  proph_Temp['ds'] = df['Time']
  proph_Temp['y'] = df['Temperature']

  model_prophet_Temp = Prophet(interval_width=0.95)
  model_prophet_Temp.fit(proph_Temp)
  future_dates = model_prophet_Temp.make_future_dataframe(periods=36, freq='MS')
  forecast = model_prophet_Temp.predict(future_dates)

  ans = datetime.strptime(datetime.today().strftime("%Y-%m-%d %H:%M:%S"),"%Y-%m-%d %H:%M:%S")
  final_Temp = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]['ds']
  rmse_prophet_Temp = np.mean((forecast.loc[:, 'yhat'] - proph_Temp['y'])**2)**.5

  try:
    ans11 = int(predic(1,model_LSTM_Temp).tolist()[0])
  except:
    ans11=0
  try:
    ans22 = int(model_arima_Temp.forecast().tolist()[0])
  except:
    ans22 = 0
  try:
    ans33 = int(model_prophet_Temp.predict(pd.DataFrame({'ds':[str(ans)]}))['yhat'].tolist()[0])//120
  except:
    ans33=0
  lst = [ans11,ans22,ans33]

  if rmse_lstm_Temp<rmse_arima_Temp and rmse_lstm_Temp<rmse_prophet_Temp:
      indt = 0

  if rmse_arima_Temp<rmse_lstm_Temp and rmse_arima_Temp<rmse_prophet_Temp:
      indt = 1

  if rmse_prophet_Temp<rmse_arima_Temp and rmse_prophet_Temp<rmse_lstm_Temp:
      indt = 2

  if (lst[indt]!=0 or lst[indt]!=0.00) and (lst[indt]>20 or lst[indt]>20.00):
    tempe = lst[indt]
  else:
    lst.pop(indt)
    if lst[0]!=0 or lst[0]!=0.00:
      tempe = lst[0]
    else:
      tempe = lst[1]


  #ARIMA
  #Oxygen Saturation
  stepwise_fit_OS = auto_arima(df['Oxygen Saturation'], trace=True,suppress_warnings=True)

  model_arima_OS=ARIMA(train['Oxygen Saturation'],order=stepwise_fit_OS.get_params().get("order"))
  model_arima_OS=model_arima_OS.fit()

  start=len(train)
  end=len(train)+len(test)-1

  pred=model_arima_OS.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')

  rmse_arima_OS = np.mean((pred - test['Oxygen Saturation'])**2)**.5

  #LSTM
  #Oxygen Saturation
  X, Y = sampling(df['Oxygen Saturation'].tolist(), n_steps)
  X_train, Y_train = sampling(df['Oxygen Saturation'][:train_count+1].tolist(), n_steps)
  X_test, Y_test = sampling(df['Oxygen Saturation'][train_count+1:].tolist(), n_steps)

  model_LSTM_OS = Sequential()
  model_LSTM_OS.add(LSTM(60, activation='relu', input_shape=(3,1)))
  model_LSTM_OS.add(Dense(1))
  model_LSTM_OS.compile(optimizer='adam', loss='mse')

  X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

  model_LSTM_OS.fit(X_train, Y_train, epochs=200, verbose=0)

  X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

  ypred = model_LSTM_OS.predict(X_test, verbose=0)

  rmse_lstm_OS = np.mean((ypred - Y_test)**2)**.5

  close_data = X.reshape((-1))

  #Prophet
  #Oxygen Saturation
  proph_OS = pd.DataFrame(columns=['ds','y'])
  proph_OS['ds'] = df['Time']
  proph_OS['y'] = df['Oxygen Saturation']

  model_prophet_OS = Prophet(interval_width=0.95)
  model_prophet_OS.fit(proph_OS)
  future_dates = model_prophet_OS.make_future_dataframe(periods=36, freq='MS')
  forecast = model_prophet_OS.predict(future_dates)

  ans = datetime.strptime(datetime.today().strftime("%Y-%m-%d %H:%M:%S"),"%Y-%m-%d %H:%M:%S")
  final_OS = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]['ds']
  rmse_prophet_OS = np.mean((forecast.loc[:, 'yhat'] - proph_OS['y'])**2)**.5

  try:
    ans111 = int(predic(1,model_LSTM_OS).tolist()[0])
  except:
    ans111=0
  try:
    ans222 = int(model_arima_OS.forecast().tolist()[0])
  except:
    ans222=0
  try:
    ans333 = int(model_prophet_OS.predict(pd.DataFrame({'ds':[str(ans)]}))['yhat'].tolist()[0])//120
  except:
    ans333=0
  lso=[ans111,ans222,ans333]

  if rmse_lstm_OS<rmse_arima_OS and rmse_lstm_OS<rmse_prophet_OS:
    indo = 0

  if rmse_arima_OS<rmse_lstm_OS and rmse_arima_OS<rmse_prophet_OS:
    indo = 1

  if rmse_prophet_OS<rmse_arima_OS and rmse_prophet_OS<rmse_lstm_OS:
    indo = 2

  if (lso[indo]!=0 or lso[indo]!=0.00) and (lso[indo]>89 or lso[indo]>89.00):
    oxys = lso[indo]
  else:
    lso.pop(indo)
    if lso[0]!=0 or lso[0]!=0.00:
      oxys = lso[0]
    else:
      oxys = lso[1]

  #ARIMA
  #Blood Glucose
  stepwise_fit_BG = auto_arima(df['Blood Glucose'], trace=True,suppress_warnings=True)

  model_arima_BG=ARIMA(train['Blood Glucose'],order=stepwise_fit_BG.get_params().get("order"))
  model_arima_BG=model_arima_BG.fit()

  start=len(train)
  end=len(train)+len(test)-1

  pred=model_arima_BG.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')

  rmse_arima_BG = np.mean((pred - test['Blood Glucose'])**2)**.5

  #LSTM
  #Blood Glucose
  X, Y = sampling(df['Blood Glucose'].tolist(), n_steps)
  X_train, Y_train = sampling(df['Blood Glucose'][:train_count+1].tolist(), n_steps)
  X_test, Y_test = sampling(df['Blood Glucose'][train_count+1:].tolist(), n_steps)

  model_LSTM_BG = Sequential()
  model_LSTM_BG.add(LSTM(60, activation='relu', input_shape=(3,1)))
  model_LSTM_BG.add(Dense(1))
  model_LSTM_BG.compile(optimizer='adam', loss='mse')

  X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

  model_LSTM_BG.fit(X_train, Y_train, epochs=200, verbose=0)

  X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

  ypred = model_LSTM_BG.predict(X_test, verbose=0)

  rmse_lstm_BG = np.mean((ypred - Y_test)**2)**.5

  close_data = X.reshape((-1))

  #Prophet
  #Blood Glucose
  proph_BG = pd.DataFrame(columns=['ds','y'])
  proph_BG['ds'] = df['Time']
  proph_BG['y'] = df['Blood Glucose']

  model_prophet_BG = Prophet(interval_width=0.95)
  model_prophet_BG.fit(proph_BG)
  future_dates = model_prophet_BG.make_future_dataframe(periods=36, freq='MS')
  forecast = model_prophet_BG.predict(future_dates)

  ans = datetime.strptime(datetime.today().strftime("%Y-%m-%d %H:%M:%S"),"%Y-%m-%d %H:%M:%S")
  final_Temp = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]['ds']
  rmse_prophet_BG = np.mean((forecast.loc[:, 'yhat'] - proph_BG['y'])**2)**.5

  try:
    ans1111 = int(predic(1,model_LSTM_BG).tolist()[0])
  except:
    ans1111=0
  try:
    ans2222 = int(model_arima_BG.forecast().tolist()[0])
  except:
    ans2222=0
  try:
    ans3333 = int(model_prophet_BG.predict(pd.DataFrame({'ds':[str(ans)]}))['yhat'].tolist()[0])//120
  except:
    ans3333=0
  lsb=[ans1111,ans2222,ans3333]

  if rmse_lstm_BG<rmse_arima_BG and rmse_lstm_BG<rmse_prophet_BG:
    indb = 0

  if rmse_arima_BG<rmse_lstm_BG and rmse_arima_BG<rmse_prophet_BG:
    indb=1

  if rmse_prophet_BG<rmse_arima_BG and rmse_prophet_BG<rmse_lstm_BG:
    indb=2

  if (lsb[indb]!=0 or lsb[indb]!=0.00) and (lsb[indb]>600 or lsb[indb]>600.00):
    bloodg = lsb[indb]
  else:
    lsb.pop(indb)
    if lsb[0]!=0 or lsb[0]!=0.00:
      bloodg = lsb[0]
    else:
      bloodg = lsb[1]

  print(bp,tempe,oxys,bloodg)
  var = str(abs(bp))+','+str(abs(tempe))+','+str(abs(oxys))+','+str(abs(bloodg))
  key = datetime.now()+timedelta(minutes=2)+timedelta(seconds=30)
  key = key.strftime("%Y-%m-%d %H:%M:%S")
  diction = {key:var}
  database1.push(diction)
  # ws.send(json.dumps({'valuess':bp,'valuess2':tempe,'valuess3':oxys,'valuess4':bloodg}))
