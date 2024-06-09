# DSC_232R_Group-Project
Data Exploration on US Traffic Congestion<br/>
Dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-traffic-congestions-2016-2022

# 1. Introduction
Our group chose to analyze data relating to traffic congestion patternsin the US.  This dataset provided traffic congestion data over a 6 year period, which would allow us enough data to begin creating a prediction model.  Traffic congestion is extremely important to understand for things like urban planning and transportation efficiency, which would greatly benfit major cities like New York, Los Angeles, and even San Diego.  This project is exciting to research because it uses large amounts of real world data to hopefully find solutions to major problems that affect the entire country on a daily basis.  There is a possibility of discovering patterns or insight that could lead to actionable solutions for reducing traffic related problems.  The broader impact of having a good predictive model for this project could have several major impacts.  One of the most important would be quality of life improvements.  Many people would receive hours of their week back that could be used on anything else that they may find important in their life.  Another impact would be economic efficiency.  Many businesses rely on road transportation for people or products.  More efficient traffic would mean less time transporting stuff that may be costing them more money the longer it is in transit.  There would also be major positive environmental benfits from reducing traffic congestion.  Cars would be spending less time on the road going the same distances which would lead to lower emissions, creating a healthier environment.  These impacts are important because they help to reduce many of the major problems that most people face on a day to day basis, such as, not enough time to spend with loved ones or saving money for companies that need to reduce costs or working towards saving our planet by reducing emissions.  

# 2. Figures 

Source: 
*Figure 1:*

Source: 
*Figure 2:*

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

-  Printing all of the Columns along with first row values to look at data types and example values
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
### Train-Test Split
- The data was split into training and testing sets with an 80-20 split. The feature used was Pressure(in), and the target variable was Congestion_Speed.
### Feature Scaling
- Standard scaling was applied to the features to ensure that they have a mean of 0 and a standard deviation of 1, which is essential for some machine learning algorithms to perform optimally
### Model Training and Evaluation
- A Linear Regression model was the first to be trained on the scaled dataset. The performance of the model was evaluated using Root Mean Square Error (RMSE) for both the training and test sets
- A Random Forest Regressor was then trained with 100 estimators. The model's performance was also evaluated using RMSE for both the training and test datasets
- Finally, a Gradient Boosting Regressor with 100 estimators was trained. RMSE was calculated for both training and test sets to evaluate the model's performance
### Comparison of Model Performance
- The RMSE values for the Linear Regression, Random Forest, and Gradient Boosting models were compared to evaluate how each model generalizes to new, unseen data. The models showed similar RMSE values for both the training and test datasets, indicating good generalization and minimal overfitting

## 3.4 Model 2
- 
- 
- 

## 3.5 Model 3
- 
- 
- 

# 4. Results
## 4.1 Data Exploration 
-  
-
-

## 4.2 Preprocessing
- 
-
- 

## 4.3 Model 1
- 
- 
- 

## 4.4 Model 2
- 
- 
- 

## 4.5 Model 3
- 
- 
- 
  
# 5. Discussion
## Data Exploration 
-  
-
-

## Preprocessing
- 
-
- 

## Model 1
- 
- 
- 

## Model 2
- 
- 
- 

## Model 3
- 
- 
- 

# 6. Conclusion

# 7. Statement of Collaboration
Name: Nikolai Pastore
Title:
Contribution:

Name: Elias Garcia
Title:
Contribution:

Name: Jackie Gao
Title:
Contribution:

Name:
Title:
Contribution:
