from __future__ import print_function
# import os.path
from googleapiclient.discovery import build
#from google_auth_oauthlib.flow import InstalledAppFlow
#from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials


from google.oauth2 import service_account

from flask import Flask, render_template, session, redirect
from functools import wraps
import pymongo
import sys,errno



SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']  
SERVICE_ACCOUNT_FILE = 'keys.json'

credentials=None
credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# If modifying these scopes, delete the file token.json.


# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = '127WRHYP7G7OMhqG5WFr1nwtbHzG1HzJS6WsG_8Jxdx4'


service = build('sheets', 'v4', credentials=credentials)
# Call the Sheets API
sheet = service.spreadsheets()
result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                            range="Sheet1!A1:F300").execute()
#print(result)
values = result.get('values', [])
#print(values)
print(values[0][1])
m=len(values)
n=len(values[0])
print(m)
print(n)

print(type(values[0][1]))
# for i in range(m):
#   for j in range(n):
#     values[i][j].encode('ascii','ignore')
p=0
q=0
for i in values:
  q=0
  for val in i:
    values[p][q]=val.encode('utf-8','ignore')
    q=q+1
  p=p+1
# print(values)

# app = Flask(__name__)
app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates')
app.secret_key = b'\xcc^\x91\xea\x17-\xd0W\x03\xa7\xf8J0\xac8\xc5'

# # Database
# client = pymongo.MongoClient("mongodb+srv://test:rk250397@cluster0.50bck.mongodb.net/waterproj?retryWrites=true&w=majority")
# db = client.test
# db = client.user_login_system

# client = pymongo.MongoClient("mongodb+srv://test:rk250397@cluster0.50bck.mongodb.net/test?retryWrites=true&w=majority")
# db = client.WF



#client = pymongo.MongoClient("mongodb+srv://test:rk250397@cluster0.50bck.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
#db = client.WF

client = pymongo.MongoClient("mongodb+srv://test:rk250397@cluster0.50bck.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client.WF


# Decorators
def login_required(f):
  @wraps(f)
  def wrap(*args, **kwargs):
    if 'logged_in' in session:
      return f(*args, **kwargs)
    else:
      return redirect('/')
  
  return wrap

# Routes
from user import routes
# from user.models import User
@app.route('/')
def home():
  return render_template('home.html')

@app.route('/signup/')
def signup1():
  return render_template('signup.html')

@app.route('/login/')
def login1():
  return render_template('login.html')

@app.route('/profile/')
@login_required
def profile():
  return render_template('profile.html')
@app.route('/dashboard/')
@login_required
def dashboard():
  # try:
  #     from google.colab import drive
  #     drive.mount('/content/drive', force_remount=True)
  #     COLAB = True
  #     print("Note: using Google CoLab")
  # #     %tensorflow_version 2.x
  # except:
  #     print("Note: not using Google CoLab")
  #     COLAB = False
  # User().start_display()
  from pandas import read_csv
  import pandas as pd
  import datetime
  from sklearn import metrics
  import pickle
  import numpy as np
  #loc='csvfiles/House'+house+'.csv'
  # loc='csvfiles/House5.csv'
  # print(loc)
  # series = read_csv(loc, header=0, parse_dates=[['Date', 'Time']])
  # series.head()
  # pd.to_datetime(series['Date_Time'])
  # df=pd.DataFrame(series)

  # df=df[['Date_Time','SensorValue']]
  # df.head()


  # pd.to_datetime(series['Date_Time'])


  # pd.to_datetime(series['Date_Time'])
  
  # df=pd.DataFrame(series)

  # df=df[['Date_Time','SensorValue']]
  # df.head()

  
  # df['datetime'] = pd.to_datetime(df['Date_Time'])
  # df.head()

  # X = df['datetime'].values
  # y = df['SensorValue'].values


  #air and water
  #print(values)
  df1=pd.DataFrame(values)
  df1.columns = df1.iloc[0]
  df1 = df1.iloc[1:]
  df1['Datetime']=pd.to_datetime(df1['Date'] + ' ' + df1['Time'])
  #df1['Datetime']=df1['Dateee']+' '+ df1['Time']
  #df1['Datetime']=pd.to_datetime(df1['Date'], errors='ignore')
  # df1['Datetime']=df1['Date'].astype('datetime64[ns]')
  df1['Air Quality(PPM)'] = df1['Air Quality(PPM)'].apply(pd.to_numeric, errors='coerce')
  df1['Turbidity(NTU)'] = df1['Turbidity(NTU)'].apply(pd.to_numeric, errors='coerce')
  df1= df1.dropna()
  print("last values")
  real_time_val=df1.tail(1)
  temp=real_time_val.iloc[0]['Temperature(C)']
  humid=real_time_val.iloc[0]['Humidity(%)']
  air=real_time_val.iloc[0]['Air Quality(PPM)']
  turb=real_time_val.iloc[0]['Turbidity(NTU)']
  print(df1.info())

  #future dates array
  # X1_future=np.array(pd.to_datetime(['2021-05-03T00:00:00.000000000','2021-05-04T00:00:00.000000000','2021-05-05T00:00:00.000000000','2021-05-06T00:00:00.000000000','2021-05-07T00:00:00.000000000','2021-05-08T00:00:00.000000000','2021-05-08T00:00:00.000000000','2021-05-09T00:00:00.000000000']))
  X1_future=np.array(pd.to_datetime(['2021-05-03','2021-05-04','2021-05-05','2021-05-06','2021-05-07','2021-05-08','2021-05-08']))
  print(type(X1_future[0]))
  print(X1_future)
  X1_future = X1_future.astype("datetime64[ns]")
  #X1_future=np.array(pd.date_range(start="2021-05-03",end="2021-05-09").to_pydatetime().tolist())
  #X1_future=np.array(['2021-05-03T00:00:00.000000000','2021-05-04T00:00:00.000000000','2021-05-05T00:00:00.000000000','2021-05-06T00:00:00.000000000','2021-05-07T00:00:00.000000000','2021-05-08T00:00:00.000000000','2021-05-08T00:00:00.000000000'])
  df1= df1[['Datetime', 'Air Quality(PPM)', 'Turbidity(NTU)']]
  #df1= df1[[ 'Air Quality', 'Turbidity']]
  print(df1)


  X = df1['Datetime'].values
  y = df1['Air Quality(PPM)'].values
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

  # # Fitting Random Forest Regression to the dataset
  # #air quality
  from sklearn.ensemble import RandomForestRegressor
  model = RandomForestRegressor(n_estimators = 1000, random_state = 0)
  model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
  

  ##printing whats in X_test
  # print("X_TESTTTTTTTTTT")
  # print(X_test)
  # #save the model
  
  # y_pred = model.predict(X_test.reshape(-1,1))
  # y_pred.reshape(-1)
  # print("AIR TRAINING THEN PLOTTING")
  # print(y_pred)
  # filename_air = 'saved_model_air.sav'
  # pickle.dump(model, open(filename_air, 'wb'))

  #check how saved model performs
  # filename_air = 'finalized_model_air.sav'
  # loaded_model = pickle.load(open(filename_air, 'rb'))

  #result = loaded_model.score(X_test, y_test)
  #print(result)
  filename_air = 'saved_model_air.sav'
  loaded_model = pickle.load(open(filename_air, 'rb'))
  y_pred = loaded_model.predict(X_test.reshape(-1,1))
  print("LOADED MODEL")
  y_pred.reshape(-1)
  print(y_pred)

  
  import matplotlib.pyplot as plt

  plt.scatter(X_test, y_test, color = 'red')
  plt.scatter(X_test, y_pred, color = 'green')
  plt.title('Random Forest Regression')
  plt.xlabel('Time')
  plt.ylabel('Air Sensor Values')
  plt.show()

  # print(y_pred)
  # print("Y TESTTTTTTTTTTTTTTTTTTTTT")
  # print(y_test)
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  day1=y_pred[0]
  day2=y_pred[1]
  day3=y_pred[2]
  day4=y_pred[3]
  day5=y_pred[4]
  day6=y_pred[5]
  day7=y_pred[6]
  week=[day1,day2,day3,day4,day5,day6,day7]

  #water
  X1 = df1['Datetime'].values
  y1 = df1['Turbidity(NTU)'].values
  from sklearn.model_selection import train_test_split
  X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.05)

  # Fitting Random Forest Regression to the dataset
  from sklearn.ensemble import RandomForestRegressor
  model = RandomForestRegressor(n_estimators = 1000, random_state = 0)
  model.fit(X1_train.reshape(-1,1), y1_train.reshape(-1,1))

  # filename_water = 'saved_model_water.sav'
  # pickle.dump(model, open(filename_water, 'wb'))
  # y1_pred = model.predict(X1_test.reshape(-1,1))
  # y1_pred.reshape(-1)
  # print("WATER TRAINING THEN PLOTTING")
  # print(y1_pred)

  filename_water = 'finalized_model_water.sav'
  loaded_model = pickle.load(open(filename_water, 'rb'))
  y1_pred = loaded_model.predict(X1_test.reshape(-1,1))
  y1_pred.reshape(-1)
  print(y1_pred)

  #print(y1_pred)

  #print(y1_test)

  day1=y1_pred[0]
  day2=y1_pred[1]
  day3=y1_pred[2]
  day4=y1_pred[3]
  day5=y1_pred[4]
  day6=y1_pred[5]
  day7=y1_pred[6]
  week1=[day1,day2,day3,day4,day5,day6,day7]
  import matplotlib.pyplot as pyplot
  days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
  pyplot.plot(days, week)
  pyplot.show()
 

  exp_bill=[week[0],week[1],week[2],week[3],week[4],week[5],week[6]]
  legend="Air Quality for House "
  legend1="Turbidity for House"
  bill=sum(exp_bill)
  bill=float("{:.2f}".format(bill))
  print(bill)
  return render_template('dashboard.html', labels=days, values=week, values1=week1, bill=bill,legend=legend, legend1=legend1, temp=temp, humid=humid, air=air, turb=turb)

if __name__ == '__main__':
    app.run(debug=True)
