## Online Shoppers Purchase Intention Prediction

This project focuses on analysing user browsing behaviour to predict whether an online shopper will complete a purchase during their session. The work includes <strong>data exploration, preprocessing, model training, feature selection, evaluation</strong>, and finally deploying a prediction interface through a Streamlit web application.
<br>

Streamlit App Link (may take a few seconds to start if inactive):
`https://mlda-cw1-16014-online-shoppers-intention.streamlit.app/`
How to install Steamlit app locally:
```
pip install -r requirements.txt
streamlit run online_shoppers_intention_ui_app.py
```
<br>

### Project Overview
The goal of this project is to classify whether a visitor will generate revenue  like purchase = 1, no purchase = 0. The dataset contains behavioural, technical, and session-related features collected from an e-commerce website.

Dataset source: Online Shoppers Purchasing Intention Data Set (UCI Machine Learning Repository).
- dataset link: `https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset`
<br>

##### Three machine learning algorithms were trained and compared:

- Random Forest Classifier
- XGBoost Classifier
- K-Nearest Neighbors (KNN)

<strong> Random Forest achieved the best performance and was selected for deployment. </strong>
<br>

### Dataset Description
The dataset contains 12,330 sessions and 18 attributes, categorized as:
```
- Numerical Features
- Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay.
```
<b>Categorical Features</b> are Month, VisitorType.
<b>Binary Features</b> are Weekend, Revenue (Target), and we can see the target column Revenue which indicates whether a purchase occurred.
<br> 

The dataset was split into 80% training and 20% testing with shuffling enabled. Before stranting training the model.
<br> 


#### Model Training and Feature Selection
Three models were trained with <strong>hyperparameter</strong> tuning using <strong>GridSearchCV</strong>.

- First model was <b>Random Forest Classifier</b> and we tuned the model parameter with <b>n_estimators</b> and we get the best value: 100 along with CV_Accuracy: 0.905
Feature selection was performed using embedded feature importance. Selected features included behaviourally meaningful variables such as <b>PageValues</b>, <b>ProductRelated_Duration</b>, <b>ExitRates</b>, and <b>BounceRates</b>.

- Second model was <b>XGBoost Classifier</b> and we tuned the model with n_estimators to find the best value: 100 and CV_Accuracy which is the 0.897. Feature importance ranking was used for selection.

- The last one is <b>K-Nearest Neighbors</b> and we used different approach like Features selection using Chi-Square Scaling applied and find out the best k: 11 along with CV_Accuracy: 0.876
<br> 

#### Project Structure
```
MLDA_CW1_16014/
│
├── source/
│   ├── t_models/                    # trained models
│   ├── s_features/                  # selected feature lists
│   ├── plots_graphs/                # contains graphs and plots .png format
│   └── EDA_and_Modelling.ipynb
│
├── ui_app/
│   └── online_shoppers_intention_ui_app.py
│
├── dataset/
│   └── online_shoppers_intention.csv
│
├── requirements.txt
└── README.md
```

#### To sum up

The project shows how user behaviour data from an e-commerce platform can be used to predict purchase intention with high accuracy. Random Forest provided strong performance without requiring feature scaling and offered clear interpretability through feature importance. The inclusion of a Streamlit interface makes the solution accessible for practical use and real-time experience.
