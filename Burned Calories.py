import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import plotly.express as px
import smtplib, ssl,datetime


from matplotlib import style
#style.use("seaborn")


import warnings
warnings.filterwarnings('ignore')

st.write("## Burned Calories Prediction")
st.image("https://previews.123rf.com/images/aquir/aquir2010/aquir201009191/157380783-burn-calories-stamp-square-grunge-sign-isolated-on-white-background.jpg" , use_column_width=True)
st.write("In this WebApp, you can find the burned calories in your body. The only thing you have to do is pass your parameters, such as `Age,` `Gender,` And `BMI,` etc., into this WebApp, and then you will be able to see the predicted value of kilocalories that burned in your body. Also, you can send your predicted calorie information to your mail id.")

#Going to create a slider in streamlit to enter the input parameters as a fuction.
st.sidebar.header("User input parameters : ")

def user_input_features():
    global age, bmi, duration, heart_rate, body_temp
    age = st.sidebar.slider("Age : " , 10 , 100 , 20)
    bmi = st.sidebar.slider("BMI : " , 15 , 40 , 20)
    duration = st.sidebar.slider("Duration (min) : " , 0 , 60 , 10)
    heart_rate = st.sidebar.slider("Heart Rate : " , 60 , 130 , 80)
    body_temp = st.sidebar.slider("Body Temperature (C) : " , 36 , 42 , 38)
    gender_button = st.sidebar.radio("Gender : ", ("Male" , "Female"))
    
    if gender_button == "Male":
        gender = 1
    else:
        gender = 0
    
    data_model = {
        "Age" : age,
        "BMI" : bmi,
        "Duration" : duration,
        "Heart_Rate" : heart_rate,
        "Body_Temp" : body_temp,
        "Gender" : gender,
    }

    features = pd.DataFrame(data_model, index=[0])
    data = pd.DataFrame(data_model, index=[0])
    
    return features, data

df, data = user_input_features()
st.write("---")
st.header("Your Parameters : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)
st.write(data)

#collecting the raw data and converting it into data frame using pandas module
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
#st.write(calories.head())
#st.write(exercise.head())
exercise_df = exercise.merge(calories , on = "User_ID")
#exercise_df = pd.concat([exercise, calories['Calories']],axis=1)
#st.write(exercise_df.head())
exercise_df.drop(columns = "User_ID" , inplace = True)
#st.write(exercise_df.head())
exercise_df.replace({'Gender':{'male':1,'female':0}},inplace=True)
#st.write(exercise_df.head())

#Going to seprate the data into training data and testing data
exercise_train_data , exercise_test_data = train_test_split(exercise_df , test_size = 0.1 , random_state = 1)

for data in [exercise_train_data , exercise_test_data]:         
    # adding BMI column to both training and test sets
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"] , 2)
#st.write(exercise_train_data.head())
#st.write(exercise_test_data.head())
exercise_train_data = exercise_train_data[["Age" , "BMI" , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories" , "Gender"]]
exercise_test_data = exercise_test_data[["Age" , "BMI"  , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories" ,"Gender" ]]
#st.write(exercise_train_data.head())
#st.write(exercise_test_data.head())

X_train = exercise_train_data.drop("Calories" , axis = 1)
y_train = exercise_train_data["Calories"]
#st.write(X_train.head())
#st.write(y_train.head())

X_test = exercise_test_data.drop("Calories" , axis = 1)
y_test = exercise_test_data["Calories"]
#st.write(X_test.head())
#st.write(y_test.head())

#going to use random forest regressor with training data and testing data to get the predicted outcome
random_reg = RandomForestRegressor(n_estimators = 3000 , max_features = 4 , max_depth = 7)
random_reg.fit(X_train , y_train)
random_reg_prediction = random_reg.predict(X_test)

prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)

global pred
pred = round(prediction[0] , 2)
st.write(pred , " **kilocalories**")

st.write("---")
st.header("Similar Results : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)

range = [prediction[0] - 10 , prediction[0] + 10]
ds = exercise_df[(exercise_df["Calories"] >= range[0]) & (exercise_df["Calories"] <= range[-1])]
st.write(ds.sample(5))

st.write("---")
st.header("General Information : ")

boolean_age = (exercise_df["Age"] < age).tolist()
boolean_duration = (exercise_df["Duration"] < duration).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < body_temp).tolist()
boolean_heart_rate= (exercise_df["Heart_Rate"] < heart_rate).tolist()

st.write("You are older than %" , round(sum(boolean_age) / len(boolean_age) , 2) * 100 , "of other people.")
st.write("Your had higher exercise duration than %" , round(sum(boolean_duration) / len(boolean_duration) , 2) * 100 , "of other people.")
st.write("You had more heart rate than %" , round(sum(boolean_heart_rate) / len(boolean_heart_rate) , 2) * 100 , "of other people during exercise.")
st.write("You had higher body temperature  than %" , round(sum(boolean_body_temp) / len(boolean_body_temp) , 2) * 100 , "of other people during exercise.")


st.write("---")
st.write("## If you want to send your details to your mail :")
st.write("Please enter your mail id :")
st.write("Note : After entering your mail id please click send.")
global email
email = st.text_input("Email :")
dt = datetime.datetime.now()


# Predicted burned calories will be sent to your mail id using the SMTP module
def SendMail():
    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls
    sender_email = "mvsramireddy1376@gmail.com"
    password = "luvujbrepineoocp"
    context = ssl.create_default_context()
    try:
        
        server = smtplib.SMTP(smtp_server,port)
        server.ehlo() # Can be omitted
        server.starttls(context=context) # Secure the connection
        server.ehlo() # Can be omitted
        server.login(sender_email, password)
        message = f"""From: sender_email
Subject: Burned Calorie Prediction Notification
\nThis is the notification of your burned calories prediction.
\nYour parameters:
\n{df}
\nYou burnt {pred} kilocalories on {dt}. 
\nThanks for using our application.
"""
        server.sendmail(sender_email, email, message )
    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        server.quit()

#print(dt)

if st.button('Send',on_click=SendMail):
    st.write("Email sent !")
    st.write("## thank you !")