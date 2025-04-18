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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

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

X = gas.iloc[:,0:1]
Y = gas.iloc[:,1:2] # A1 

fig, ax = plt.subplots()
ax.scatter(X, Y)
ax.set_title('A1 before Prediction')
ax.set_xlabel('X')
ax.set_ylabel('Y')

st.pyplot(fig)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.45, random_state=42)

# Convert the 'Date' column to ordinal to avoid DTypePromotionError
X_train['Date'] = X_train['Date'].apply(lambda x: x.toordinal())
X_test['Date'] = X_test['Date'].apply(lambda x: x.toordinal())

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

X_test['Date'] = X_test['Date'].apply(lambda x: datetime.fromordinal(x))  # Convert back to datetime for display

fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color='b', label='Actual')
ax.plot(X_test, y_pred, color='k', label='Predicted')

# # Annotate each point with a number starting from 0
# for i, txt in enumerate(range(len(X_test))):
#     ax.annotate(txt, (X_test.iloc[i], y_test.iloc[i]),arrowprops = dict(facecolor='yellow',shrink=0.09)) # to see which one number to annotate for later  

    
ax.annotate("2007-2008 Great Recession", (X_test.iloc[684],y_test.iloc[684]),xytext=(-150,-40),textcoords='offset points', bbox=dict(boxstyle='round',fc='0.8'),arrowprops= dict(arrowstyle='->'))

ax.annotate("-2011 Civil war in Libya\n -Higher-than-normal refinery \nmaintenance on the\nGulf and East Coasts", (X_test.iloc[607],y_test.iloc[607]),xytext=(-80,100),textcoords='offset points', bbox=dict(boxstyle='round',fc='0.8'),arrowprops= dict(arrowstyle='->'))

ax.annotate("-2009 recession effects \n Lower Demand from High Unemployment \n Slowed Production(Factories & Power-Plants)", (X_test.iloc[265],y_test.iloc[265]),xytext=(-80,-50),textcoords='offset points', bbox=dict(boxstyle='round',fc='0.8'),arrowprops= dict(arrowstyle='->'))

ax.annotate("-2020 Global Pandemic", (X_test.iloc[334],y_test.iloc[334]),xytext=(50,-50),textcoords='offset points', bbox=dict(boxstyle='round',fc='0.8'),arrowprops= dict(arrowstyle='->'))

ax.annotate("-2016 Oil Production greatly increased", (X_test.iloc[648],y_test.iloc[648]),xytext=(70,50),textcoords='offset points', bbox=dict(boxstyle='round',fc='0.8'),arrowprops= dict(arrowstyle='->'))

ax.annotate("-2022 Pandemic Recovery\n-Gasoline Demand Rises \n Gasoline inventory low\n -Russia Invasion of Ukraine", (X_test.iloc[173],y_test.iloc[173]),xytext=(50,-50),textcoords='offset points', bbox=dict(boxstyle='round',fc='0.8'),arrowprops= dict(arrowstyle='->'))

plt.title("A1 Gas: Weekly U.S. All Grades All Formulations Retail Gasoline Prices")
plt.xlabel('Year')
plt.ylabel('US dollar per Gallon')
plt.legend()

st.pyplot(fig)


st.write('Mean Squared Error:', metrics.mean_squared_error(y_test.values, y_pred))
# Convert the 'Date' column to ordinal to avoid DTypePromotionError
X_test['Date'] = X_test['Date'].apply(lambda x: x.toordinal())

# Calculate the score
score = reg_all.score(X_test, y_test)

# Convert 'Date' column back to datetime for further use
X_test['Date'] = X_test['Date'].apply(lambda x: datetime.fromordinal(x))

st.write("A1-R^2" , score * 100)

# Initialize polynomial features, scaler, and ridge regression model for typeOF other than 'L'.
poly = PolynomialFeatures(degree=2)
scaler = StandardScaler()
poly_ridge = Ridge(alpha=1.0)

# Add selectbox for regression type
regression_type = st.selectbox(
    "Select Regression Type",
    ("Linear Regression", "Polynomial Ridge Regression")
)
def A1Predict(date_str,typeOF):
    date_to_predict = datetime.strptime(date_str, '%Y-%m-%d').toordinal()
    if(typeOF == 'L'):
        predicted_value = reg_all.predict(np.array([[date_to_predict]]))
        predicted_date = datetime.fromordinal(date_to_predict)
        return predicted_value[0][0]
    else: 
        predicted_value = poly_ridge.predict(scaler.fit_transform(poly.fit_transform(np.array([[date_to_predict]]))))
        predicted_date = datetime.fromordinal(date_to_predict)

        return predicted_value[0][0]
        
# Modify A1 prediction logic based on selected regression type
if regression_type == "Linear Regression":
    st.write("Using Linear Regression for A1")
    # Linear Regression code for A1
    from datetime import datetime, timedelta



    # Initialize start and end dates
    start_date = datetime(2024, 8, 19)
    end_date = datetime(2029, 8, 19)  # Corrected end date

    # Create a list to store the dates
    date_list = []

    # Generate dates weekly
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(weeks=1)

    # Convert the list to a DataFrame
    date_df = pd.DataFrame(date_list, columns=['Date'])

    # Convert the 'Date' column to string format for prediction
    date_df['Date'] = date_df['Date'].dt.strftime('%Y-%m-%d')

    # Apply the A1Predict function to each row
    date_df['A1 Prediction'] = date_df['Date'].apply(A1Predict,typeOF='L')

    fig, ax = plt.subplots()

    date_df.plot(x='Date', y='A1 Prediction', ax=ax)

    ax.set_xlabel('Date')
    ax.set_ylabel('Predicted Gas Prices')
    ax.set_title('A1: Linear Prediction Model (5 years)')
    ax.tick_params(axis='x', rotation=90)
    st.pyplot(fig)


else:
    st.write("Using Polynomial Ridge Regression for A1")
    # Polynomial Ridge Regression code for A1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.45, random_state=42)

    # Shape of Training set and Testing set
   # print('Original set  ---> ', X.shape, Y.shape, '\nTraining set  ---> ', X_train.shape, y_train.shape, '\nTesting set   ---> ', X_test.shape, y_test.shape)

    # Convert the 'Date' column to ordinal to avoid DTypePromotionError
    X_train['Date'] = X_train['Date'].apply(lambda x: x.toordinal())
    X_test['Date'] = X_test['Date'].apply(lambda x: x.toordinal())

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    poly_ridge = Ridge()
    poly_ridge.fit(X_train_poly, y_train)
    poly_ridge_predictions = poly_ridge.predict(X_test_poly)

    # Calculate the score
    score = poly_ridge.score(X_test_poly, y_test)

    # Convert 'Date' column back to datetime for further use
    X_test['Date'] = X_test['Date'].apply(lambda x: datetime.fromordinal(x))

    # Scatter plot
    fig, ax = plt.subplots()

    ax.scatter(X_test['Date'], y_test, color='b', label='Actual')
    ax.scatter(X_test['Date'], poly_ridge_predictions, color='g', label='Predicted')

    ax.set_xlabel('Date')
    ax.set_ylabel('US dollar per Gallon')
    ax.set_title('A1 Polynomial Ridge Regression Predictions vs Actual values')
    ax.legend()
    ax.tick_params(axis='x', rotation=90)

    st.pyplot(fig)


    # Display the results
    
   
Y_R1 = gas.iloc[:,3:4] # R1 
X_R1= gas.iloc[:,0:1]

#print(Y)
    # Assuming X and Y are already defined
X_train_R1, X_test_R1, y_train_R1, y_test_R1 = train_test_split(X_R1, Y_R1, test_size=0.45, random_state=42)

    # Convert the 'Date' column to ordinal to avoid DTypePromotionError
X_train_R1['Date'] = X_train_R1['Date'].apply(lambda x: x.toordinal())
X_test_R1['Date'] = X_test_R1['Date'].apply(lambda x: x.toordinal())

reg_all_R1 = LinearRegression()
reg_all_R1.fit(X_train_R1, y_train_R1)
y_pred_R1= reg_all.predict(X_test_R1)

X_test_R1['Date'] = X_test_R1['Date'].apply(lambda x: datetime.fromordinal(x))  # Convert back to datetime for display   
if regression_type == "Linear Regression":
    st.write("Using Linear Regression for R1")
    # Linear Regression code for R1
    # R1

  

    fig, ax = plt.subplots()

    ax.scatter(X_test_R1, y_test_R1, color='b')
    ax.plot(X_test_R1, y_pred_R1, color='k')

    ax.set_title("R1: Weekly U.S. Regular All Formulations Retail Gasoline Prices")
    ax.set_xlabel('Year')
    ax.set_ylabel('US dollar per Gallon')


    st.pyplot(fig)
    st.write('Mean Squared Error:', metrics.mean_squared_error(y_test_R1, y_pred_R1))
    # Convert the 'Date' column to ordinal to avoid DTypePromotionError
    X_test_R1['Date'] = X_test_R1['Date'].apply(lambda x: x.toordinal())

    # Calculate the score
    score = reg_all.score(X_test_R1, y_test_R1)

    # Convert 'Date' column back to datetime for further use
    X_test_R1['Date'] = X_test_R1['Date'].apply(lambda x: datetime.fromordinal(x))

    st.write('R^2' , score * 100)
    
else:
    st.write("Using Polynomial Ridge Regression for R1")
    # Polynomial Ridge Regression code for R1
        # POLY R1
# Split the data into training and testing sets
    X_train_R1, X_test_R1, y_train_R1, y_test_R1 = train_test_split(X_R1, Y_R1, test_size=0.45, random_state=42)

    # Shape of Training set and Testing set
   # print('Original set  ---> ', X_R1.shape, Y_R1.shape, '\nTraining set  ---> ', X_train_R1.shape, y_train_R1.shape, '\nTesting set   ---> ', X_test_R1.shape,y_test_R1.shape)

    # Convert the 'Date' column to ordinal to avoid DTypePromotionError
    X_train_R1['Date'] = X_train_R1['Date'].apply(lambda x: x.toordinal())
    X_test_R1['Date'] = X_test_R1['Date'].apply(lambda x: x.toordinal())

    # Standardize the data
    scaler = StandardScaler()
    X_train_R1_scaled = scaler.fit_transform(X_train_R1)
    X_test_R1_scaled = scaler.transform(X_test_R1)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_train_R1_poly = poly.fit_transform(X_train_R1_scaled)
    X_test_R1_poly = poly.transform(X_test_R1_scaled)

    poly_ridge_R1 = Ridge()
    poly_ridge_R1.fit(X_train_poly, y_train)
    poly_ridge_predictions_R1 = poly_ridge_R1.predict(X_test_R1_poly)

    # Calculate the score
    score_R1 = poly_ridge_R1.score(X_test_R1_poly, y_test_R1)

    # Convert 'Date' column back to datetime for further use
    X_test_R1['Date'] = X_test_R1['Date'].apply(lambda x: datetime.fromordinal(x))

    # Scatter plot
    fig, ax = plt.subplots()

    ax.scatter(X_test_R1['Date'], y_test_R1, color='b', label='Actual')
    ax.scatter(X_test_R1['Date'], poly_ridge_predictions_R1, color='g', label='Predicted')

    ax.set_xlabel('Date')
    ax.set_ylabel('US dollar per Gallon')
    ax.set_title('R1 Polynomial Ridge Regression Predictions vs Actual values')
    ax.legend()
    st.pyplot(fig)

    # Display the results
    st.write('Accuracy:',score_R1* 100)
    st.write('Mean Squared Error:', metrics.mean_squared_error(y_test_R1, y_pred_R1))



# M1
Y_M1 = gas.iloc[:,2:3] # R1 
X_M1= gas.iloc[:,0:1]

# print(Y)
# Assuming X and Y are already defined
X_train_M1, X_test_M1, y_train_M1, y_test_M1 = train_test_split(X_M1, Y_M1, test_size=0.45, random_state=42)

# Convert the 'Date' column to ordinal to avoid DTypePromotionError
X_train_M1['Date'] = X_train_M1['Date'].apply(lambda x: x.toordinal())
X_test_M1['Date'] = X_test_M1['Date'].apply(lambda x: x.toordinal())

reg_all_M1 = LinearRegression()
reg_all_M1.fit(X_train_M1, y_train_M1)
y_pred_M1= reg_all.predict(X_test_M1)

X_test_M1['Date'] = X_test_M1['Date'].apply(lambda x: datetime.fromordinal(x))  # Convert back to datetime for display

if regression_type == "Linear Regression":
    st.write("Using Linear Regression for M1")
    # Linear Regression code for M1
    fig, ax = plt.subplots()
    ax.scatter(X_test_M1, y_test_M1, color ='b') 
    ax.plot(X_test_M1, y_pred_M1, color ='k') 
    ax.set_title('M1: Weekly U.S. Midgrade All Formulations Retail Gasoline Prices')
    ax.set_xlabel('Year')
    ax.set_ylabel('US dollar per Gallon')
    st.pyplot(fig)
    st.write('Mean Squared Error:', metrics.mean_squared_error(y_test_M1, y_pred_M1))
# Convert the 'Date' column to ordinal to avoid DTypePromotionError
    X_test_M1['Date'] = X_test_M1['Date'].apply(lambda x: x.toordinal())

    # Calculate the score
    score = reg_all.score(X_test_M1, y_test_M1)

    # Convert 'Date' column back to datetime for further use
    X_test_M1['Date'] = X_test_M1['Date'].apply(lambda x: datetime.fromordinal(x))

    st.write('M1 R^2' , score * 100)
else:
    st.write("Using Polynomial Ridge Regression for M1")
    # Polynomial Ridge Regression code for M1
    X_train_M1, X_test_M1, y_train_M1, y_test_M1 = train_test_split(X_M1, Y_M1, test_size=0.45, random_state=42)

# Shape of Training set and Testing set
    #print('Original set  ---> ', X_M1.shape, Y_M1.shape, '\nTraining set  ---> ', X_train_M1.shape, y_train_M1.shape, '\nTesting set   ---> ', X_test_M1.shape,y_test_M1.shape)

    # Convert the 'Date' column to ordinal to avoid DTypePromotionError
    X_train_M1['Date'] = X_train_M1['Date'].apply(lambda x: x.toordinal())
    X_test_M1['Date'] = X_test_M1['Date'].apply(lambda x: x.toordinal())

    # Standardize the data
    scaler = StandardScaler()
    X_train_M1_scaled = scaler.fit_transform(X_train_M1)
    X_test_M1_scaled = scaler.transform(X_test_M1)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_train_M1_poly = poly.fit_transform(X_train_M1_scaled)
    X_test_M1_poly = poly.transform(X_test_M1_scaled)

    poly_ridge_M1 = Ridge()
    poly_ridge_M1.fit(X_train_poly, y_train)
    poly_ridge_predictions_M1 = poly_ridge_M1.predict(X_test_M1_poly)

    # Calculate the score
    score_M1 = poly_ridge_M1.score(X_test_M1_poly, y_test_M1)

    # Convert 'Date' column back to datetime for further use
    X_test_M1['Date'] = X_test_M1['Date'].apply(lambda x: datetime.fromordinal(x))

    # Scatter plot
    fig, ax = plt.subplots()
    ax.scatter(X_test_M1['Date'], y_test_M1, color='b', label='Actual')
    ax.scatter(X_test_M1['Date'], poly_ridge_predictions_M1, color='g', label='Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('US dollar per Gallon')
    ax.set_title('M1 Polynomial Ridge Regression Predictions vs Actual values')
    ax.legend()
    st.pyplot(fig)

    # Display the results
    st.write('Accuracy:',score_R1* 100)
    st.write('Mean Squared Error:', metrics.mean_squared_error(y_test_M1, y_pred_M1))


Y_P1 = gas.iloc[:,5:6]
X_P1= gas.iloc[:,0:1]

# Assuming X and Y are already defined
X_train_P1, X_test_P1, y_train_P1, y_test_P1 = train_test_split(X_P1, Y_P1, test_size=0.45, random_state=42)

# Convert the 'Date' column to ordinal to avoid DTypePromotionError
X_train_P1['Date'] = X_train_P1['Date'].apply(lambda x: x.toordinal())
X_test_P1['Date'] = X_test_P1['Date'].apply(lambda x: x.toordinal())

reg_all_P1 = LinearRegression()
reg_all_P1.fit(X_train_P1, y_train_P1)
y_pred_P1= reg_all.predict(X_test_P1)

X_test_P1['Date'] = X_test_P1['Date'].apply(lambda x: datetime.fromordinal(x))  # Convert back to datetime for display

if regression_type == "Linear Regression":
    st.write("Using Linear Regression for P1")
    # Linear Regression code for P1
    fig, ax = plt.subplots()

    ax.scatter(X_test_P1, y_test_P1, color='b', label='Actual')
    ax.plot(X_test_P1, y_pred_P1, color='k', label='Predicted')

    ax.set_title('P1: Weekly U.S. Premium All Formulations Retail Gasoline Prices')
    ax.set_xlabel('Year')
    ax.set_ylabel('US dollar per Gallon')
    ax.legend()
    st.pyplot(fig)
    st.write('Mean Squared Error:', metrics.mean_squared_error(y_test_P1, y_pred_P1))
    # Convert the 'Date' column to ordinal to avoid DTypePromotionError
    X_test_P1['Date'] = X_test_P1['Date'].apply(lambda x: x.toordinal())

    # Calculate the score
    score = reg_all.score(X_test_P1, y_test_P1)

    # Convert 'Date' column back to datetime for further use
    X_test_P1['Date'] = X_test_P1['Date'].apply(lambda x: datetime.fromordinal(x))

    st.write('P1-R^2:' , score * 100)


else:
    st.write("Using Polynomial Ridge Regression for P1")
    # Polynomial Ridge Regression code for 
    X_train_P1, X_test_P1, y_train_P1, y_test_P1 = train_test_split(X_P1, Y_P1, test_size=0.45, random_state=42)

    # Shape of Training set and Testing set
    #print('Original set  ---> ', X_P1.shape, Y_P1.shape, '\nTraining set  ---> ', X_train_P1.shape, y_train_P1.shape, '\nTesting set   ---> ', X_test_P1.shape,y_test_P1.shape)

    # Convert the 'Date' column to ordinal to avoid DTypePromotionError
    X_train_P1['Date'] = X_train_P1['Date'].apply(lambda x: x.toordinal())
    X_test_P1['Date'] = X_test_P1['Date'].apply(lambda x: x.toordinal())

    # Standardize the data
    scaler = StandardScaler()
    X_train_P1_scaled = scaler.fit_transform(X_train_P1)
    X_test_P1_scaled = scaler.transform(X_test_P1)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_train_P1_poly = poly.fit_transform(X_train_P1_scaled)
    X_test_P1_poly = poly.transform(X_test_P1_scaled)

    poly_ridge_P1 = Ridge()
    poly_ridge_P1.fit(X_train_P1_poly, y_train_P1)
    poly_ridge_predictions_P1 = poly_ridge_P1.predict(X_test_P1_poly)

    # Calculate the score
    score_P1 = poly_ridge_P1.score(X_test_P1_poly, y_test_P1)

    # Convert 'Date' column back to datetime for further use
    X_test_P1['Date'] = X_test_P1['Date'].apply(lambda x: datetime.fromordinal(x))

    # Scatter plot
    fig, ax = plt.subplots()

    ax.scatter(X_test_P1['Date'], y_test_P1, color='b', label='Actual')
    ax.scatter(X_test_P1['Date'], poly_ridge_predictions_P1, color='g', label='Predicted')

    ax.set_xlabel('Date')
    ax.set_ylabel('US dollar per Gallon')
    ax.set_title('P1: Polynomial Ridge Regression Predictions vs Actual values')
    ax.legend()
    st.pyplot(fig)


    # Display the results
    st.write('Accuracy:',score_P1* 100)
    st.write('Mean Squared Error:', metrics.mean_squared_error(y_test_P1, poly_ridge_predictions_P1))
    
Y_D1 = gas.iloc[:,4:5] # D1

X_D1= gas.iloc[:,0:1] # date



# Assuming X and Y are already defined
X_train_D1, X_test_D1, y_train_D1, y_test_D1 = train_test_split(X_D1, Y_D1, test_size=0.2, random_state=42)

# Convert the 'Date' column to ordinal to avoid DTypePromotionError
X_train_D1['Date'] = X_train_D1['Date'].apply(lambda x: x.toordinal())
X_test_D1['Date'] = X_test_D1['Date'].apply(lambda x: x.toordinal())

reg_all_D1 = LinearRegression()
reg_all_D1.fit(X_train_D1, y_train_D1)
y_pred_D1= reg_all.predict(X_test_D1)

X_test_D1['Date'] = X_test_D1['Date'].apply(lambda x: datetime.fromordinal(x))  # Convert back to datetime for display

if regression_type == "Linear Regression":
    st.write("Using Linear Regression for D1")
    # Linear Regression code for D1
    fig, ax = plt.subplots()

    ax.scatter(X_test_D1, y_test_D1, color='b', label='Actual')
    ax.plot(X_test_D1, y_pred_D1, color='k', label='Predicted')

    ax.set_xlabel('Year')
    ax.set_ylabel('US dollar per Gallon')
    ax.set_title('D1: Weekly U.S. No 2 Diesel Retail Prices')
    ax.legend()

    st.pyplot(fig)
    st.write('Mean Squared Error:', metrics.mean_squared_error(y_test_D1, y_pred_D1))
# Convert the 'Date' column to ordinal to avoid DTypePromotionError
    X_test_D1['Date'] = X_test_D1['Date'].apply(lambda x: x.toordinal())

    # Calculate the score
    score = reg_all.score(X_test_D1, y_test_D1)

    # Convert 'Date' column back to datetime for further use
    X_test_D1['Date'] = X_test_D1['Date'].apply(lambda x: datetime.fromordinal(x))

    st.write('D1 R^2' , score * 100)

else:
    st.write("Using Polynomial Ridge Regression for D1")
    # Polynomial Ridge Regression code for D1
    X_train_D1, X_test_D1, y_train_D1, y_test_D1 = train_test_split(X_D1, Y_D1, test_size=0.3, random_state=42)


# Shape of Training set and Testing set
    # print('Original set  ---> ', X_D1.shape, Y_D1.shape, '\nTraining set  ---> ', X_train_D1.shape, y_train_D1.shape, '\nTesting set   ---> ', X_test_D1.shape,y_test_D1.shape)

    # Convert the 'Date' column to ordinal to avoid DTypePromotionError
    X_train_D1['Date'] = X_train_D1['Date'].apply(lambda x: x.toordinal())
    X_test_D1['Date'] = X_test_D1['Date'].apply(lambda x: x.toordinal())

    # Standardize the data
    scaler = StandardScaler()
    X_train_D1_scaled = scaler.fit_transform(X_train_D1)
    X_test_D1_scaled = scaler.transform(X_test_D1)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_train_D1_poly = poly.fit_transform(X_train_D1_scaled)
    X_test_D1_poly = poly.transform(X_test_D1_scaled)

    poly_ridge_D1 = Ridge()
    poly_ridge_D1.fit(X_train_D1_poly, y_train_D1)
    poly_ridge_predictions_D1 = poly_ridge_D1.predict(X_test_D1_poly)

    # Calculate the score
    score_D1 = poly_ridge_D1.score(X_test_D1_poly, y_test_D1)

    # Convert 'Date' column back to datetime for further use
    X_test_D1['Date'] = X_test_D1['Date'].apply(lambda x: datetime.fromordinal(x))

    # Scatter plot
    fig, ax = plt.subplots()

    ax.scatter(X_test_D1['Date'], y_test_D1, color='b', label='Actual')
    ax.scatter(X_test_D1['Date'], poly_ridge_predictions_D1, color='g', label='Predicted')

    ax.set_xlabel('Date')
    ax.set_ylabel('US dollar per Gallon')
    ax.set_title('D1 Polynomial Ridge Regression Predictions vs Actual values')
    ax.legend()
    st.pyplot(fig)

    # Display the results
    st.write('Accuracy:', score_D1 * 100)
    st.write('Mean Squared Error:', metrics.mean_squared_error(y_test_D1, poly_ridge_predictions_D1))

