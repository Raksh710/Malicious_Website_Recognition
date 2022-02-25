# Malicious_Website_Recognition
Classifying Malicious website from benign ones using CatBoost Classifier. 
The data was heavily imbalanced with 88% bias towards benign class (Type=0) and only 12% samples had Type=1 or Malicious website.
Process involves Exploration of data, Data Cleaning, Resampling of data (to handle highly imbalanced data), Model implementation and Evaluation.
Catboost classifier turned out to be the most robust model giving us approximately the appropriate values for Recall and F-1 score (since we were supposed to focus on recognizing 'malicious' website more than benign ones).
#### **Recall: 0.7894**
#### **F-1 Score: 0.7126**
