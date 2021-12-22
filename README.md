# Heart Failure Prediction Classification: Project Overview
This project uses multiple logistic regression to predict heart failure rate
* Create a tool that predicts the rate of heart failure to help people with heart condition assess and predict their heart condition
* Use Logistic Regression and Random Forest Classifier
## Code and Resources Used
* **Python Version:** 3.7
* **Packages:** pandas, numpy, matplotlib, seaborn, scikit learn
* **Kaggle:** https://www.kaggle.com/fedesoriano/heart-failure-prediction
## About the data
The data contain the following columns:
* Age: age of the patient [years]
* Sex: sex of the patient [M: Male, F: Female]
* ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
* RestingBP: resting blood pressure [mm Hg]
* Cholesterol: serum cholesterol [mm/dl]
* FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
* RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
* MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
* ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
* Oldpeak: oldpeak = ST [Numeric value measured in depression]
* ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
* HeartDisease: output class [1: heart disease, 0: Normal]
## Data Cleaning
I needed to clean the data up so that it was usable for our model. I made the followinng changes and created the following variables:
* Replaced 0 values in cholesterol with the median
* Removed rows with 0 value in restingECG
* Transformed categorical data into dummies valuables
## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables.

![alt text](https://github.com/Panasak/Heart_Failure_Prediction_Classification_Project/blob/main/EDA/age.png)
![alt text](https://github.com/Panasak/Heart_Failure_Prediction_Classification_Project/blob/main/EDA/chestpaintype.png)
![alt text](https://github.com/Panasak/Heart_Failure_Prediction_Classification_Project/blob/main/EDA/restingecg.png)
## Model Building
First I transformed the categorical variables into dummy variables. I also split the data into train and test sets with a test size of 30%
I tired two different models and evaluated them using Classification Report (Precision, Recall, F-1 score, Accuracy Score). 
I tried two different models:
* **Logistic Regression** - Baseline for the model
* **Random Forest** - Because of the sparse data from the many categorical variables. I thought a normalized regression like rfc would be effective
## Model Performance
The Logistic Regression model outperformed the other apprach slightly on the test and validation sets
* **Linear Regression:** Accuracy score = 0.87, precision = 0.93, recall = 0.88, f-1 score = 0.90
* **Random Forest:** Accuracy score = 0.88, precision = 0.90, recall = 0.89, f-1 score = 0.89
