import streamlit as st
import requests 
import pandas as pd
import csv 
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

st.title("Analyzing Gasoline Price Trends: A Predictive Approach” \n By Christian Minchala, Kevin Heneson , and Hao Ye ")
st.markdown("[DataCamp Project] https://www.datacamp.com/datalab/w/6b841625-a072-44f2-889b-7e94a76a088b/edit")
st.markdown("### Data Notes")
st.info("No need to data clean because the data is already clean.")

st.markdown("### Data Contents")
fuel_info = [
    "A1: Weekly U.S. All Grades All Formulations Retail Gasoline Prices (Dollars per Gallon)",
    "A2: Weekly U.S. All Grades Conventional Retail Gasoline Prices (Dollars per Gallon)",
    "A3: Weekly U.S. All Grades Reformulated Retail Gasoline Prices (Dollars per Gallon)",
    "R1: Weekly U.S. Regular All Formulations Retail Gasoline Prices (Dollars per Gallon)",
    "R2: Weekly U.S. Regular Conventional Retail Gasoline Prices (Dollars per Gallon)",
    "R3: Weekly U.S. Regular Reformulated Retail Gasoline Prices (Dollars per Gallon)",
    "M1: Weekly U.S. Midgrade All Formulations Retail Gasoline Prices (Dollars per Gallon)",
    "M2: Weekly U.S. Midgrade Conventional Retail Gasoline Prices (Dollars per Gallon)",
    "M3: Weekly U.S. Midgrade Reformulated Retail Gasoline Prices (Dollars per Gallon)",
    "P1: Weekly U.S. Premium All Formulations Retail Gasoline Prices (Dollars per Gallon)",
    "P2: Weekly U.S. Premium Conventional Retail Gasoline Prices (Dollars per Gallon)",
    "P3: Weekly U.S. Premium Reformulated Retail Gasoline Prices (Dollars per Gallon)",
    "D1: Weekly U.S. No 2 Diesel Retail Prices (Dollars per Gallon)",
]

for item in fuel_info:
    st.markdown(f"- {item}")

st.caption("Source: U.S. Energy Information Administration (Aug 2024)")
st.info(" We will be focusing on only A1,R1,M1,P1,D1")

All_grade = pd.read_csv("Weekly_U.S._All_Grades_All_Formulations_Retail_Gasoline_Prices.csv",parse_dates=['Date']) #A1
Midgrade = pd.read_csv('Weekly_U.S._Midgrade_All_Formulations_Retail_Gasoline_Prices.csv',parse_dates=['Date'])# M1
Premium_grade = pd.read_csv('Weekly_U.S._Premium_All_Formulations_Retail_Gasoline_Prices.csv',parse_dates=['Date'])# P1
Regular = pd.read_csv('Weekly_U.S._Regular_All_Formulations_Retail_Gasoline_Prices.csv',parse_dates=['Date'])# R1
Diesel = pd.read_csv('Weekly_U.S._No_2_Diesel_Retail_Prices.csv',parse_dates=['Date']) # D1

# Values of csv
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

cols = st.columns(2, gap="large")


with cols[0]:
    st.markdown("### - Data of A1")
    st.write(All_grade_subset)

with cols[1]:
    st.markdown("### -Data for M1")
    st.write(Midgrade)


cols2 = st.columns(2)

with cols2[0]:
    st.markdown("### Data for R1 Regular")
    st.write(Regular_subset)
with cols2[1]:
    st.markdown("### Data for P1 Premium ")
    st.write(Premium_grade_subset)

st.markdown("### Data for D1 Diesel")
st.write(Diesel_subset)

gas = All_grade_subset.merge(Midgrade, on='Date', how='inner') \
                            .merge(Regular_subset, on='Date', how='inner') \
                            .merge(Diesel_subset, on='Date', how='inner') \
                            .merge(Premium_grade_subset, on='Date', how='inner')

st.markdown("### Merging the dataframes on the 'Date' column ")
st.write(gas)
# Heatmap
fig,ax = plt.subplots()
sns.heatmap(gas.corr(), annot = True)
st.pyplot(fig)


# Distribution 
fig, ax = plt.subplots()

ax.hist(gas['A1'])
ax.set_title('Distribution of All Grade Gasoline Prices')
ax.set_xlabel('Price')
ax.set_ylabel('Count')


st.pyplot(fig)


#Gasoline price over time

fig, ax = plt.subplots()

ax.plot(gas['Date'], gas['A1'], label='All Grades')
ax.plot(gas['Date'], gas['M1'], label='Mid-grade')
ax.plot(gas['Date'], gas['P1'], label='Premium Grade')
ax.plot(gas['Date'], gas['D1'], label='Diesel')
ax.plot(gas['Date'], gas['R1'], label='Regular grade')

ax.set_title('Gasoline Prices Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()

st.pyplot(fig)
fig = px.line(x=gas['Date'], y=gas['A1'], labels={'y':'All Grades'}, template='plotly_dark')
st.plotly_chart(fig)




# cols = st.columns(5)

# cols[0].write("hi sssssssssssssssssssss")
# cols[1].write("hi sssssssssssssssssssss")
# cols[2].write("hi sssssssssssssssssssss")
# cols[3].write("hi sssssssssssssssssssss")
# cols[4].write("hi")
