# In-Hospital-Mortality-Prediction
### Problem Statement-
In hospital mortality prediction for patients remains a challenge for healthcare systems.
A great percent of patients at the hospital end up losing their life either because of lack of resources to understand and handle their case or for that matter understand the intensity of the situation to treat the patient.
Being able to understand the in-hospital mortality statistics is necessary to understand the quality of care provided, accurately predicting prognosis and receiving intensive treatment.

### Purpose
Risk assessment of in-hospital mortality of patients at the time of hospitalization is necessary for determining the scale of required medical resources for the patient depending on the patient’s severity. 
Applying ML technique to this issue can lead to an accurate prediction model and it can be used by the healthcare systems to save lives of many.
It can help healthcare workers to prioritize assessing severity of illness and adjudicating the value of novel treatments, interventions, and health care policies.

### Data analysis workflow
Our ML Model implementation follows this workflow. Details on each step are elaborated in this document.

![](https://github.com/schopde1/In-Hospital-Mortality-Prediction/blob/main/Images/Flow%20Diagram.png)

### Dataset at a Glance

##### Source : [Dataset](https://www.kaggle.com/saurabhshahane/in-hospital-mortality-prediction (Shahane, 2021)) 

##### Data collection:  Demographic characteristics and vital signs extracted were recorded during the ﬁrst 24 hours of each admission and laboratory variables were measured during the entire ICU stay. Comorbidities were identified using ICD-9 codes. 

Demographic characteristics - age at the time of hospital admission, sex, ethnicity, weight, and height. 

Vital signs - heart rate, (HR), systolic blood pressure [SBP], diastolic blood pressure [DBP], mean blood pressure, respiratory rate, body temperature, saturation pulse oxygen [SPO2], urine output [first 24 h].

Comorbidities - hypertension, atrial fibrillation, ischemic heart disease, diabetes mellitus, depression, hypoferric anemia, hyperlipidemia, chronic kidney disease (CKD), and chronic obstructive pulmonary disease [COPD].

Laboratory variables - hematocrit, red blood cells, mean corpuscular hemoglobin [MCH], mean corpuscular hemoglobin concentration [MCHC], mean corpuscular volume [MCV], red blood cell distribution width [RDW], platelet count, white blood cells, neutrophils, basophils, lymphocytes, prothrombin time [PT], international normalized ratio [INR], NT-proBNP, creatine kinase, creatinine, blood urea nitrogen [BUN] glucose, potassium, sodium, calcium, chloride, magnesium, the anion gap, bicarbonate, lactate, hydrogen ion concentration [pH], partial pressure of CO2 in arterial blood, and LVEF.


##### Shape : 1177 rows x 51 columns



##### Dependent variable : Outcome (Alive or Dead) Alive-0, Dead-1

### Data Preparation Summary

##### Cleaning the data: 
* Checking Null values.
* Using SimpleImputer to replace NAN with mean values.


##### Visualization
* Relation of each predictor with the outcome before and after PCA.
* Principal components selected using variance.
* Data Imbalance vs Data Balance post SMOTE
* Confusion Matrix for different Models.
* Plot to measure Model Accuracy.


##### Scaling and Dimension reduction
* Used StandardScaler to standardize date
* Used PCA to reduce predictors
* Visualized the PC components


##### Splitting the data set
* Used Train-test split- 70% Training,30% Test.


##### Handling data imbalance
* Used SMOTE to balance data


### Data Preparation- Details

##### Cleaning the data
Verified the dataset for null or missing values.

SimpleImputer which is a scikit-learn class, which is helpful in handling the missing data in the predictive model dataset. It replaces the NaN values with a specified placeholder. The strategy we used is MEAN which replaces the NaN with mean values along each column.


##### Scaling
In the machine learning algorithms if the values of the features are closer to each other there are chances for the algorithm to get trained well and faster but if they have high value difference it can take more time to understand the data and the accuracy will be lower. So, if the data in any conditions has data points far from each other, scaling is a technique to make them closer to each other.   

We used StandardScaler to scale our datasets.


##### Principal Component Analysis
We used PCA by setting the variance of the input and as a result, we get the compressed representation of the input data. By doing PCA ,the number of components reduced from 51 to 36 with 95% cutoff threshold.

PCA is a statistical procedure that converts a set of observations of possibly correlated variables into a set of    values of linearly uncorrelated variables called principal components.


##### Handling Data imbalance
Post Visualizing the data, we inferred that the data was highly imbalanced. We implemented SMOTE, which is an approach to address imbalanced dataset by oversampling the minority class.

SMOTE is a type of data augmentation that synthesizes new samples from the existing ones. Yes — SMOTE creates “new” data points by duplicating existing ones.






 
### VISUALIZATIONS CREATED DURING DATA PREPARATION
Used Seaborn which is a data visualization library based on matplotlib we implemented some visualizations to view our data. 

Relationship of a few predictors with the outcome-

![](https://github.com/schopde1/In-Hospital-Mortality-Prediction/blob/main/Images/Relation.png)

PCA Threshold variance and PCA relation with the outcome

![](https://github.com/schopde1/In-Hospital-Mortality-Prediction/blob/main/Images/Threshold.png)


Data Imbalance- Before and After SMOTE

![Before SMOTE](https://github.com/schopde1/In-Hospital-Mortality-Prediction/blob/main/Images/Beforesmote.png) 
![After SMOTE](https://github.com/schopde1/In-Hospital-Mortality-Prediction/blob/main/Images/Aftersmote.png)


### MODELS IMPLEMENTED ON THE DATASET
Algorithms towards diagnosis or forecasting (prognosis) of an event were based on supervised learning to predict mortality using preprocessed data and several ML approaches. The ML algorithms we used were logistic regression, random forest, K-nearest neighbors, and XGBoost. We performed analysis in Python using the Scikit learn package. 
 
#### Logistic Regression
LR is a statistical method for analyzing datasets, and it is also a supervised machine-learning algorithm developed for learning classification problems.It is one of the most widely used methods in health sciences research, especially in epidemiology.Some studies have shown that LR is effective in analyzing mortality factors and predicting mortality.
##### Accuracy obtained from Logistic regression - 0.69

#### K-Nearest Neighbour
The K-Nearest Neighbour or KNN is an algorithm that stores all available instances and classifies new instances based on a similarity measure (such as distance functions).It has been widely used in classification and regression prediction problems owing to its simple implementation and outstanding performance. We feel that for our dataset this model can help identify patients mortality based on other patients with similar parameters.
##### Accuracy obtained from KNN- 0.69

#### Random Forest
RF is an ensemble supervised machine learning algorithm. It uses a decision tree as the base classifier. RF produces many classifiers and combines their results through majority voting.Since ours is a classification problem this model will be well suited.
##### Accuracy obtained from Random forest- 0.87

#### XGBoost
XGBoost is an implementation of gradient boosted decision trees designed for speed and performance that is dominative competitive machine learning. It helps boost the accuracy of model prediction.This is particularly suited for our dataset because we have all numerical values and outcome is binary.
##### Accuracy of XGBoost model- 0.90

### Confusion Matrix for each model
##### Logistic Regression
![](https://github.com/schopde1/In-Hospital-Mortality-Prediction/blob/main/Images/LRCF.png)

##### K-Nearest Neighbour
![](https://github.com/schopde1/In-Hospital-Mortality-Prediction/blob/main/Images/KNCF.png)


##### XGBoost 
![](https://github.com/schopde1/In-Hospital-Mortality-Prediction/blob/main/Images/XGCM.png)


##### Random Forest
![](https://github.com/schopde1/In-Hospital-Mortality-Prediction/blob/main/Images/KNCF.png)


### MODELS EVALUATION- BEST MODEL VISUALIZATION
We implemented 4 ML models on our dataset and found that the Random Forest model has the best accuracy and it outperformed all the other models we implemented. The boosting model XGBoost also had a good performance and came very close to the accuracy of Random Forest model.  We used a boxplot diagram to show the model which has the best accuracy.

![](https://github.com/schopde1/In-Hospital-Mortality-Prediction/blob/main/Images/Result.png)


### RECOMMENDATIONS
The data set was created from a diverse population with a wide variety of life-threatening conditions when they were admitted at the hospital. The data set had frequent missing and occasionally incorrect observations. We found the data set to be highly unbalanced and we expected this challenge to be difficult.

In this study, we developed a model that predicts in-hospital mortality with high predictive performance using machine learning technology and variables of age, sex, BMI, Lab and blood sampling test results. This machine learning model has the possibility to be useful in evaluating the in-hospital mortality risk of admitted patients. 

We found that Random Forest was the top performing and had the best accuracy in terms of predicting the outcome than the rest of the models. But, when it comes to Hospitals and Healthcare systems, we have no room for error and hence, we would like to explore more in terms of Machine learning, Artificial Intelligence, Algorithm development to make this model more accurate and contribute our little bit towards saving lives of many.

