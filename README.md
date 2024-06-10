# DSC_232R_Group-Project
Data Exploration on US Traffic Congestion<br/>
Dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-traffic-congestions-2016-2022

# 1. Introduction
Our group chose to analyze data relating to traffic congestion patternsin the US.  This dataset provided traffic congestion data over a 6 year period, which would allow us enough data to begin creating a prediction model.  Traffic congestion is extremely important to understand for things like urban planning and transportation efficiency, which would greatly benfit major cities like New York, Los Angeles, and even San Diego.  This project is exciting to research because it uses large amounts of real world data to hopefully find solutions to major problems that affect the entire country on a daily basis.  There is a possibility of discovering patterns or insight that could lead to actionable solutions for reducing traffic related problems.  The broader impact of having a good predictive model for this project could have several major impacts.  One of the most important would be quality of life improvements.  Many people would receive hours of their week back that could be used on anything else that they may find important in their life.  Another impact would be economic efficiency.  Many businesses rely on road transportation for people or products.  More efficient traffic would mean less time transporting stuff that may be costing them more money the longer it is in transit.  There would also be major positive environmental benfits from reducing traffic congestion.  Cars would be spending less time on the road going the same distances which would lead to lower emissions, creating a healthier environment.  These impacts are important because they help to reduce many of the major problems that most people face on a day to day basis, such as, not enough time to spend with loved ones or saving money for companies that need to reduce costs or working towards saving our planet by reducing emissions.  

# 2. Figures  
*Figure 1: Correlation Map*
![image](https://github.com/EliasGarciaa/DSC_232R_Group-Project/assets/29644567/3509f6fe-a6c8-4afc-b2c7-2c051fb141cd)


*Figure 2: Severity and distribution of traffic delays*
![image](https://github.com/EliasGarciaa/DSC_232R_Group-Project/assets/29644567/ef679903-d5ab-416e-9635-37b56d2f07e5)


# 3. Methods
## 3.1 Data Exploration 
The dataset used in the analysis includes information on weather and traffic over a 6 year period in the United States.  The dataset is obtained from [kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-traffic-congestions-2016-2022).  For ease of access, it is replicated in this repository.  

The dataset consists of the following columns:

1. ID - Unique Identifier for data<br/>
2. Severity - Ranges 0 to 4 - Severity of event<br/>
3. DelayFromFreeFlowSpeed(mins) - Delay compared to free traffic flow (in minutes) due to congestion event<br/>
4. StartTime - Approximate time of congestion event, given in their local timezone<br/>
5. Congestion_Speed - Slow, Moderate, Fast - Speed of traffic impacted by congestion<br/>
6. State - Location of where this event happened<br/>
7. Pressure(in) - Air pressure (in inches)<br/>
8. Visiblity(mi) - Visiblity (in miles)<br/>
9. WindSpeed(mph) - Speed of the wind (in miles per hour)<br/>
10. Precipitation(in) - Rain amount (in inches)<br/>
11. Weather_Event - Weather event such as rain, snow, hail, thunderstorm, etc.<br/>

The data exploration consists of several parts:

- Printing all of the Columns along with first row values to look at data types and example values
- Printing the Schema of the dataset to obtain a complete list of features
- Defining the shape of the dataset by printing the number of Columns and Rows

## 3.2 Preprocessing
- Missing values are handled by being dropped from the main dataframe
- Duplicates are removed from the main dataframe
- Specific fields/features are filtered into lists
- Columns are cleaned and updated by adding filtering expressions
- Features are grouped together for dimensionality reduction and gathering domain-specific knowledge
- Efficient data storage, performance, schema evolution, interoperability, and cost efficiency is ensured by creating Parquet files

## 3.3 Model 1
### Data Preprocessing
- The data preprocessing phase involved loading preprocessed datasets and merging relevant columns from two dataframes: accident_df and weather_df. Rows with missing values were removed to ensure a complete dataset for model training
```
# Load the preprocessed data
accident_df = pd.read_parquet(os.getcwd() + "/parquet_data/accident_df.parquet")
weather_df = pd.read_parquet(os.getcwd() + "/parquet_data/weather_table.parquet")

# Merge relevant columns from accident_df and weather_df
merged_df = accident_df[['ID', 'Congestion_Speed']].merge(
    weather_df[['ID', 'WeatherTimeStamp', 'Pressure(in)']], on='ID', how='inner')

# Drop rows with missing values
merged_df.dropna(inplace=True)
```
### Train-Test Split
- The data was split into training and testing sets with an 80-20 split. The feature used was Pressure(in), and the target variable was Congestion_Speed.
```
from sklearn.model_selection import train_test_split

# Splitting the data into train and test sets
X = merged_df[['Pressure(in)']]
y = merged_df['Congestion_Speed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### Feature Scaling
- Standard scaling was applied to the features to ensure that they have a mean of 0 and a standard deviation of 1, which is essential for some machine learning algorithms to perform optimally
```
from sklearn.preprocessing import StandardScaler

# Preprocessing: Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
### Model Training and Evaluation
- A Linear Regression model was the first to be trained on the scaled dataset. The performance of the model was evaluated using Root Mean Square Error (RMSE) for both the training and test sets
```
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Train the first model: Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
train_preds = model.predict(X_train_scaled)
test_preds = model.predict(X_test_scaled)

# Evaluation: Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
```
- A Random Forest Regressor was then trained with 100 estimators. The model's performance was also evaluated using RMSE for both the training and test datasets
```
from sklearn.ensemble import RandomForestRegressor

# Training the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predictions
rf_train_preds = rf_model.predict(X_train_scaled)
rf_test_preds = rf_model.predict(X_test_scaled)

# Evaluation: Calculate RMSE
rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_preds))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_preds))
```
- Finally, a Gradient Boosting Regressor with 100 estimators was trained. RMSE was calculated for both training and test sets to evaluate the model's performance
```
from sklearn.ensemble import GradientBoostingRegressor

# Train the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Predictions
gb_train_preds = gb_model.predict(X_train_scaled)
gb_test_preds = gb_model.predict(X_test_scaled)

# Evaluation: Calculate RMSE
gb_train_rmse = np.sqrt(mean_squared_error(y_train, gb_train_preds))
gb_test_rmse = np.sqrt(mean_squared_error(y_test, gb_test_preds))
```
### Comparison of Model Performance
- The RMSE values for the Linear Regression, Random Forest, and Gradient Boosting models were compared to evaluate how each model generalizes to new, unseen data. The models showed similar RMSE values for both the training and test datasets, indicating good generalization and minimal overfitting
```
# RMSE values for the models
models = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
train_rmse = [1.1714642543584903, 1.1683265209258846, 1.1686818156484327]
test_rmse = [1.1699850428341243, 1.166988589658763, 1.1671017369772374]
```
## 3.4 Model 2
### Data Preprocessing
- The data preprocessing phase involved loading preprocessed datasets and merging relevant columns from the dataframes: accident_df, testing_df, location_df, and weather_df.
```
accident_df = pd.read_parquet(os.getcwd() + "/parquet_data/accident_df.parquet")
weather_df = pd.read_parquet(os.getcwd() + "/parquet_data/weather_table.parquet")
testing_df = pd.read_parquet(os.getcwd() + "/parquet_data/testing_df.parquet")
location_df = pd.read_parquet(os.getcwd() + "/parquet_data/location_table.parquet")

intermediate_df = pd.merge(accident_df[["ID", "Severity", "Start_Lat", "Start_Lng","StartTime"]],weather_df[["ID", "Weather_Conditions"]], on="ID" , how = "inner")
intermediate_df = pd.merge(location_df[["ID", "ZipCode", "LocalTimeZone"]], intermediate_df, on = "ID" , how = "inner")
merged_df = pd.merge(intermediate_df, testing_df[["ID", "Visibility(mi)", "Congestion_Speed"]], on ="ID", how = "inner")
```
- One hot encoding was used to convert the categorical 'Weather Condition' Column into binary encoding for future processing
```
#encode weather conditions using one hot encoding
encoder = OneHotEncoder(sparse_output=False)  # Use sparse=True to return a sparse matrix

# Fit and transform the data
encoded_data = encoder.fit_transform(merged_df[['Weather_Conditions']])

# Create a DataFrame with the encoded data
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Weather_Conditions']))
```
- The 'StartTime' column was transformed into a datetime object, and the hours were extracted into a separate column called StartHour
```
merged_df['StartTime'] = pd.to_datetime(merged_df['StartTime'])
merged_df['StartHour'] = merged_df['StartTime'].dt.hour
```
### Feature Scaling
- PCA was applied to the one-hot encoded weather condition data to reduce its dimensionality and generate compact vector representations for subsequent processing
```
merged_df['StartTime'] = pd.to_datetime(merged_df['StartTime'])
merged_df['StartHour'] = merged_df['StartTime'].dt.hour
```
### Train-Test Split
- The dataset was divided into training and testing sets, with 80% allocated to training and 20% to testing. The features included Severity, Start_Lat, Start_Lng, StartHour, and PCA-transformed components. The target variable was Congestion_Speed.

### Model Training and Evaluation
- An XGBoost model was employed to classify the congestion speed (slow,moderate,fast) using the transformed features. The model's performance was assessed by plotting the error against the number of boosting rounds. Additionally, feature importance analysis was conducted to understand the contribution of each feature to the model's predictions
####Model Building
```
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Convert the data into DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'multi:softmax',  # Multiclass classification
    'num_class': 3,  # Number of classes (0, 1, 2)
    'eta': 0.3,
    'max_depth': 8,
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.8,
    'reg_lambda': 1.0,              # L2 regularization term on weights
    'eval_metric': 'merror'  # Classification error
}
num_round = 1000
evals = [(dtrain, 'train'), (dtest, 'eval')]
evals_result = {}  # Dictionary to store evaluation results

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=num_round, evals=evals, evals_result=evals_result, verbose_eval=False)

# Predict on the test set
preds = bst.predict(dtest)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy}")

# Print classification report
print(classification_report(y_test, preds))
```
####Plot Error rate
```
train_error = evals_result['train']['merror']
val_error = evals_result['eval']['merror']

train_accuracy = [error for error in train_error]
val_accuracy = [error for error in val_error]

# Plotting the error
plt.figure(figsize=(10, 6))
plt.plot(train_accuracy, label='Train Error')
plt.plot(val_accuracy, label='Validation Error')
plt.xlabel('Number of Boosting Rounds')
plt.ylabel('Error')
plt.title('XGBoost Error over Boosting Rounds')
plt.legend()
plt.grid(True)
plt.show()
```
####Plot feature importance
```
from xgboost import plot_importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plot_importance(bst, max_num_features=10)
plt.title('Feature Importance')
plt.show()
```
# 4. Results
## 4.3 Model 1

## 4.4 Model 2
- 
- 
- 
  
# 5. Discussion
## Data Exploration 
-  Using this data set to determine severity and other important factors, such as delays due to traffic conditions, can be very insightful as we drive daily. This data set had information on all weather conditions, such as ongoing weather events, visibility, wind speed, and even precipitation. We wanted to use features that were easily measurable and had enough data to impact the models we were using significantly. In addition to weather data, the dataset included detailed information such as the time and location of the events and the amount of time it delayed their usual commute. Analyzing these aspects allowed us to identify patterns and correlations between weather conditions and traffic severity. For instance, we could determine if certain weather conditions led to more severe accidents or longer delays.

## Preprocessing
- After thoroughly examining our dataset, we identified several preprocessing steps to ensure data quality and relevance. First, we handled missing values by dropping the Weather_Event column, as 94% of its entries were null, rendering it unusable for meaningful analysis. We also addressed redundancy by removing duplicate records, which helped maintain the dataset’s integrity and accuracy. We excluded columns such as city, county, state, and country since the latitude and longitude columns already provided precise geographical information. Additionally, we processed the StartTime column to retain only the hour of occurrence. This transformation was aimed at analyzing the impact of different times of the day on traffic patterns, enabling us to identify peak traffic periods and understand how the time of day influences traffic flow. For efficient storage and retrieval, we used Parquet files, which lead to lower query and memory costs, especially since we had a large amount of data. These preprocessing steps ensured our dataset was clean, relevant, and formatted for accurate analysis, laying a solid foundation for subsequent analysis and model building.

## Model 1
- 
- 
- 

## Model 2
- 
- 
- 

# 6. Conclusion
-

# 7. Statement of Collaboration
Name: Nikolai Pastore<br/>
Title: Teammate 1<br/>
Contribution: <br/>

Name: Elias Garcia<br/>
Title: Teammate 2<br/>
Contribution: <br/>

Name: Jackie Gao<br/>
Title: Teammate 3<br/>
Contribution: <br/>

Name: Daniel Lan<br/>
Title: Teammate 4<br/>
Contribution: <br/>
