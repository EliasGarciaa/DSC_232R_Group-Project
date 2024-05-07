# DSC_232R_Group-Project
Data Exploration on US Traffic Congestion

## Dataset Used
Traffic Congestion Data that covers fourty-nine states within 2016 and 2022: <br/>
https://www.kaggle.com/datasets/sobhanmoosavi/us-traffic-congestions-2016-2022  <br/>

Features used:  <br/>
ID - Unique Identifier for data<br/>
Severity - Ranges 0 to 4 - Severity of event<br/>
DelayFromFreeFlowSpeed(mins) - Delay compared to free traffic flow (in minutes) due to congestion event<br/>
StartTime - Approximate time of congestion event, given in their local timezone<br/>
Congestion_Speed - Slow, Moderate, Fast - Speed of traffic impacted by congestion<br/>
State - Location of where this event happened<br/>
Pressure(in) - Air pressure (in inches)<br/>
Visiblity(mi) - Visiblity (in miles)<br/>
WindSpeed(mph) - Speed of the wind (in miles per hour)<br/>
Precipitation(in) - Rain amount (in inches)<br/>
Weather_Event - Weather event such as rain, snow, hail, thunderstorm, etc.<br/>

## Environment Setup Instructions
Recommended to run on 10 cores and 10 gb/core

## Preprocessing Data
Preprocessing involves multiple steps to (1) clean, (2) transform, and (3) prepare the "US Traffic Congestions (2016-2022)" Dataset: 

### 1. Importing Libraries
### 2. Creating a Spark Session
### 3. Loading Data
### 4. Exploring Data
### 5. Data Cleaning
### 6. Feature Engineering
### 7. Data Transformation
### 8. Scaling and Normalization
### 9. Splitting Data
### 10. (Optional) Persist Data
