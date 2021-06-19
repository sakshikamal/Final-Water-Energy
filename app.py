from flask import Flask, render_template, session, redirect
from functools import wraps
import pymongo

app = Flask(__name__)
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
  # loc='csvfiles/House'+house+'.csv'
  loc='csvfiles/House5.csv'
  print(loc)
  series = read_csv(loc, header=0, parse_dates=[['Date', 'Time']])
  series.head()
  pd.to_datetime(series['Date_Time'])
  df=pd.DataFrame(series)

  df=df[['Date_Time','SensorValue']]
  df.head()

  

  pd.to_datetime(series['Date_Time'])

  

  pd.to_datetime(series['Date_Time'])



  
  df=pd.DataFrame(series)

  df=df[['Date_Time','SensorValue']]
  df.head()

  
  df['datetime'] = pd.to_datetime(df['Date_Time'])
  df.head()

  X = df['datetime'].values
  y = df['SensorValue'].values

  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

  # Fitting Random Forest Regression to the dataset
  from sklearn.ensemble import RandomForestRegressor
  model = RandomForestRegressor(n_estimators = 10, random_state = 0)
  model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

  y_pred = model.predict(X_test.reshape(-1,1))
  y_pred.reshape(-1);
  print(y_pred)

  import numpy as np
  import matplotlib.pyplot as plt

  # plt.scatter(X_test, y_test, color = 'red')
  # plt.scatter(X_test, y_pred, color = 'green')
  # plt.title('Random Forest Regression')
  # plt.xlabel('Time')
  # plt.ylabel('Sensor Values')
  # plt.show()

  print(y_pred)

  print(y_test)

  day1=y_pred[0]/60
  day2=y_pred[1]/60
  day3=y_pred[2]/60
  day4=y_pred[3]/60
  day5=y_pred[4]/60
  day6=y_pred[5]/60
  day7=y_pred[6]/60
  week=[day1,day2,day3,day4,day5,day6,day7]

  import matplotlib.pyplot as pyplot
  days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
  #pyplot.plot(days, week)
  #pyplot.show()

  exp_bill=[6*week[0],6*week[1],6*week[2],6*week[3],6*week[4],6*week[5],6*week[6]]
  legend="Electricity usage for House "
  
  bill=sum(exp_bill)
  bill=float("{:.2f}".format(bill))
  print(bill)
  return render_template('dashboard.html', labels=days, values=week, bill=bill,legend=legend)


@app.route("/dashboard/API_key=<api_key>/mac=<mac>/field=<int:field>/temp=<temp>&humid=<humid>&air=<air>&turb=<turb>", methods=['GET'])
def update(api_key, mac, field, temp, humid, air, turb):
  from pandas import read_csv
  import pandas as pd
  import datetime
  # loc='csvfiles/House'+house+'.csv'
  loc='csvfiles/House5.csv'
  print(loc)
  series = read_csv(loc, header=0, parse_dates=[['Date', 'Time']])
  series.head()
  pd.to_datetime(series['Date_Time'])
  df=pd.DataFrame(series)

  df=df[['Date_Time','SensorValue']]
  df.head()

  

  pd.to_datetime(series['Date_Time'])

  

  pd.to_datetime(series['Date_Time'])



  
  df=pd.DataFrame(series)

  df=df[['Date_Time','SensorValue']]
  df.head()

  
  df['datetime'] = pd.to_datetime(df['Date_Time'])
  df.head()

  X = df['datetime'].values
  y = df['SensorValue'].values

  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

  # Fitting Random Forest Regression to the dataset
  from sklearn.ensemble import RandomForestRegressor
  model = RandomForestRegressor(n_estimators = 10, random_state = 0)
  model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

  y_pred = model.predict(X_test.reshape(-1,1))
  y_pred.reshape(-1);
  print(y_pred)

  import numpy as np
  import matplotlib.pyplot as plt

  # plt.scatter(X_test, y_test, color = 'red')
  # plt.scatter(X_test, y_pred, color = 'green')
  # plt.title('Random Forest Regression')
  # plt.xlabel('Time')
  # plt.ylabel('Sensor Values')
  # plt.show()

  print(y_pred)

  print(y_test)

  day1=y_pred[0]/60
  day2=y_pred[1]/60
  day3=y_pred[2]/60
  day4=y_pred[3]/60
  day5=y_pred[4]/60
  day6=y_pred[5]/60
  day7=y_pred[6]/60
  week=[day1,day2,day3,day4,day5,day6,day7]

  import matplotlib.pyplot as pyplot
  days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
  #pyplot.plot(days, week)
  #pyplot.show()

  exp_bill=[6*week[0],6*week[1],6*week[2],6*week[3],6*week[4],6*week[5],6*week[6]]
  legend="Electricity usage for House "
  
  bill=sum(exp_bill)
  bill=float("{:.2f}".format(bill))
  print(bill)
  return render_template('dashboard.html', labels=days, values=week, bill=bill,legend=legend,temp=temp, humid=humid, air=air, turb=turb)
  # return render_template("update.html", data=data)
# app.run(host='0.0.0.0', port= 8090)