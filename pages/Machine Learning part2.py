import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import copy
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
st.title("Machine learning with different algorithms")
# Values of csv
All_grade = pd.read_csv("Weekly_U.S._All_Grades_All_Formulations_Retail_Gasoline_Prices.csv",parse_dates=['Date']) #A1
Midgrade = pd.read_csv('Weekly_U.S._Midgrade_All_Formulations_Retail_Gasoline_Prices.csv',parse_dates=['Date'])# M1
Premium_grade = pd.read_csv('Weekly_U.S._Premium_All_Formulations_Retail_Gasoline_Prices.csv',parse_dates=['Date'])# P1
Regular = pd.read_csv('Weekly_U.S._Regular_All_Formulations_Retail_Gasoline_Prices.csv',parse_dates=['Date'])# R1
Diesel = pd.read_csv('Weekly_U.S._No_2_Diesel_Retail_Prices.csv',parse_dates=['Date']) # D1

All_grade = All_grade.sort_values(by = 'Date')
All_grade_subset = All_grade[All_grade['Date'] >= '1994-11-28']

Midgrade = Midgrade.sort_values(by='Date')

Regular = Regular.sort_values(by='Date')

Regular_subset = Regular[Regular['Date'] >='1994-11-28']

Premium_grade = Premium_grade.sort_values(by='Date')

Premium_grade_subset = Premium_grade[Premium_grade['Date'] >='1994-11-28']

Premium_grade = Premium_grade.sort_values(by='Date')

Premium_grade_subset = Premium_grade[Premium_grade['Date'] >='1994-11-28']
Diesel = Diesel.sort_values(by='Date')

Diesel_subset = Diesel[Diesel['Date'] >='1994-11-28']
gas = All_grade_subset.merge(Midgrade, on='Date', how='inner') \
                            .merge(Regular_subset, on='Date', how='inner') \
                            .merge(Diesel_subset, on='Date', how='inner') \
                            .merge(Premium_grade_subset, on='Date', how='inner')


newGas = copy.copy(gas)

newGas = newGas.set_index("Date")
newGas = newGas.sort_index()
newGas = newGas.asfreq('W-MON')  # Set frequency to weekly, starting on Monday
newGas = newGas.sort_index(ascending=True)
st.write(newGas)

start_date = newGas.index.min()
end_date = newGas.index.max()
complete_date_range = pd.date_range(start=start_date, end=end_date, freq=newGas.index.freq)
is_index_complete = (newGas.index == complete_date_range).all()


steps = 250
data_train = newGas[-steps:]   
data_test  = newGas[:-steps]
print(
    f"Train dates : {data_train.index.min()} --- "
    f"{data_train.index.max()}  (n={len(data_train)})"
)
print(
    f"Test dates  : {data_test.index.min()} --- "
    f"{data_test.index.max()}  (n={len(data_test)})"
)

fig, ax = plt.subplots(figsize=(12, 6))
data_train['A1'].plot(ax=ax, label='train')
data_test['A1'].plot(ax=ax, label='test')
ax.set_title('Train on Late data')
ax.legend()
st.pyplot(fig)

from xgboost import XGBRegressor
forecaster = ForecasterAutoreg(
                 regressor = RandomForestRegressor(random_state=123),
                 lags = 6,
             )

forecaster_gb= ForecasterAutoreg(regressor = XGBRegressor(random_state=123),
                                lags=6,
                                )
forecaster.fit(y=data_train['A1'])

forecaster_gb.fit(y=data_train['A1'])
# st.write(forecaster)
# st.write(forecaster_gb)

steps = 35 # Predicts 35 week into future 
predictions = forecaster.predict(steps=steps)

st.markdown("## Predictions with Random Forest Regressor")
st.write(predictions)

st.markdown("## Predictions with Gradient Boosting")
predictionGB= forecaster_gb.predict(steps=steps)
st.write(predictionGB)

predictions.index = pd.date_range(start=data_train.index[-1], periods=steps, freq='W-MON')

# Plot only the predictions with weekly dates
fig, ax = plt.subplots(figsize=(12, 6))
predictions.plot(ax=ax, label='Random Forest Regressor Predictions', linestyle='--')
ax.legend()
ax.set_title('Weekly A1 Predictions')
ax.set_xlabel('Date')
ax.set_ylabel('A1')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(12, 6))

predictionGB.plot(ax=ax, label='Gradient Boosting', color='red')
predictions.plot(ax=ax, label='Random Forest Regressor Predictions', linestyle='--')

ax.set_title('A1 Gradient Boosting Predictions')
ax.legend()
st.pyplot(fig)

forecasterM1 = ForecasterAutoreg(
                 regressor = RandomForestRegressor(random_state=123),
                 lags = 6
             )
forecasterM1.fit(y=data_train['M1'])
forecasterM1

steps = 36 # Predicts 35 week into future 
predictionsM1 = forecasterM1.predict(steps=steps)
predictionsM1
predictionsM1.index = pd.date_range(start=data_train.index[-1], periods=steps, freq='W-MON')

# Plot only the predictions with weekly dates
fig, ax = plt.subplots(figsize=(12, 6))
predictionsM1.plot(ax=ax, label='Random Forest Regressor Predictions', linestyle='--')
ax.legend()
ax.set_title('Weekly M1 Predictions')
ax.set_xlabel('Date')
ax.set_ylabel('R1')
st.pyplot(fig)


forecasterR1 = ForecasterAutoreg(
                 regressor = RandomForestRegressor(random_state=123),
                 lags = 6
             )
forecasterR1.fit(y=data_train['R1'])
# forecasterR1

steps = 36 # Predicts 35 week into future 
predictionsR1 = forecasterR1.predict(steps=steps)
# predictionsR1

predictionsR1.index = pd.date_range(start=data_train.index[-1], periods=steps, freq='W-MON')

# Plot only the predictions with weekly dates
fig, ax = plt.subplots(figsize=(12, 6))
predictionsR1.plot(ax=ax, label='Random Forest Regressor Predictions', linestyle='--')
ax.legend()
ax.set_title('Weekly R1 Predictions')
ax.set_xlabel('Date')
ax.set_ylabel('R1')
st.pyplot(fig)

forecasterD1 = ForecasterAutoreg(
                 regressor = RandomForestRegressor(random_state=123),
                 lags = 6
             )
forecasterD1.fit(y=data_train['D1'])


steps = 36 # Predicts 36 week into future 
predictionsD1 = forecasterD1.predict(steps=steps)

fig, ax = plt.subplots(figsize=(12, 6))
predictionsD1.plot(ax=ax, label='Random Forest Regressor Predictions', linestyle='--')
ax.legend()
ax.set_title('Weekly Diesel D1 Predictions')
ax.set_xlabel('Date')
ax.set_ylabel('D1')
st.pyplot(fig)


forecasterP1 = ForecasterAutoreg(
                 regressor = RandomForestRegressor(random_state=1),
                 lags = 6
             )
forecasterP1.fit(y=data_train['P1'])
forecasterP1

steps = 36 # Predicts 35 week into future 
predictionsP1 = forecasterP1.predict(steps=steps)


# Generate predictionsP1 with a datetime index


fig, ax = plt.subplots(figsize=(12, 6))

# Plot the new predictions with weekly dates
predictionsP1.plot(ax=ax, label='Random Forest Regressor Predictions', linestyle='--', color='r')

# Customize the plot
ax.set_title('P1 Predictions Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('P1 Prediction')
ax.tick_params(axis='x', rotation=45)
ax.grid(True)
ax.legend()
fig.tight_layout()
st.pyplot(fig)


# Late data

steps = 250
data_train_E = newGas[:-steps]   
data_test_E  = newGas[-steps:]
print(
    f"Train dates : {data_train_E.index.min()} --- "
    f"{data_train.index.max()}  (n={len(data_train)})"
)
print(
    f"Test dates  : {data_test_E.index.min()} --- "
    f"{data_test.index.max()}  (n={len(data_test)})"
)

fig, ax = plt.subplots(figsize=(12, 6))
data_train_E['A1'].plot(ax=ax, label='train')
data_test_E['A1'].plot(ax=ax, label='test')
ax.set_title('Train on Late data')
ax.legend()
st.pyplot(fig)

forecasterA1E = ForecasterAutoreg(
                 regressor = RandomForestRegressor(random_state=1),
                 lags = 6
             )
forecasterA1E.fit(y=data_train_E['A1'])

forecasterM1E = ForecasterAutoreg(
                 regressor = RandomForestRegressor(random_state=1),
                 lags = 6
             )
forecasterM1E.fit(y=data_train_E['M1'])

forecasterR1E = ForecasterAutoreg(
                 regressor = RandomForestRegressor(random_state=1),
                 lags = 6
             )
forecasterR1E.fit(y=data_train_E['R1'])

forecasterP1E = ForecasterAutoreg(
                 regressor = RandomForestRegressor(random_state=1),
                 lags = 6
             )
forecasterP1E.fit(y=data_train_E['P1'])

forecasterD1E = ForecasterAutoreg(
                 regressor = RandomForestRegressor(random_state=1),
                 lags = 6
             )
forecasterD1E.fit(y=data_train_E['D1'])


steps = 275 # Predicts 35 week into future 
predictionsA1E = forecasterA1E.predict(steps=steps)
predictionsA1E= predictionsA1E[245:275]


steps = 275 # Predicts 35 week into future 
predictionsM1E = forecasterM1E.predict(steps=steps)
predictionsM1E= predictionsM1E[245:275]


steps = 275 # Predicts 35 week into future 
predictionsR1E = forecasterR1E.predict(steps=steps)
predictionsR1E= predictionsR1E[245:275]

steps = 275 # Predicts 35 week into future 
predictionsP1E = forecasterP1E.predict(steps=steps)
predictionsP1E= predictionsP1E[245:275]

steps = 275 # Predicts 35 week into future 
predictionsD1E = forecasterD1E.predict(steps=steps)
predictionsD1E= predictionsD1E[245:275]

fig, axs = plt.subplots(5, 1, figsize=(12, 20))

# Plot predictionsA1E
axs[0].plot(predictionsA1E, label='A1 Predictions', color='blue')
axs[0].set_title('A1 Predictions')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('A1')
axs[0].legend()

# Plot predictionsM1E
axs[1].plot(predictionsM1E, label='M1 Predictions', color='green')
axs[1].set_title('M1 Predictions')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('M1')
axs[1].legend()

# Plot predictionsR1E
axs[2].plot(predictionsR1E, label='R1 Predictions', color='red')
axs[2].set_title('R1 Predictions')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('R1')
axs[2].legend()

# Plot predictionsP1E
axs[3].plot(predictionsP1E, label='P1 Predictions', color='purple')
axs[3].set_title('P1 Predictions')
axs[3].set_xlabel('Time')
axs[3].set_ylabel('P1')
axs[3].legend()

# Plot predictionsD1E
axs[4].plot(predictionsD1E, label='D1 Predictions', color='orange')
axs[4].set_title('D1 Predictions')
axs[4].set_xlabel('Time')
axs[4].set_ylabel('D1')
axs[4].legend()

# Adjust layout
fig.tight_layout()
st.pyplot(fig)