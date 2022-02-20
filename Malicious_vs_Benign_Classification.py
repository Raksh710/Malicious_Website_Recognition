#!/usr/bin/env python
# coding: utf-8

# ## AIM:
# The objective is to classify Malicious (Type=1) websites from Benign (Type=0) ones on the basis of various features given in the dataset.<br>
# Target variable: `"TYPE"`

# ## Features:
# `URL`: it is the anonimous identification of the URL analyzed in the study <br>
# `URL_LENGTH`: it is the number of characters in the URL <br>
# `NUMBERSPECIALCHARACTERS`: it is number of special characters identified in the URL, such as, “/”, “%”, “#”, “&”, “. “, “=” <br>
# `CHARSET`: it is a categorical value and its meaning is the character encoding standard (also called character set). <br>
# `SERVER`: it is a categorical value and its meaning is the operative system of the server got from the packet response. <br>
# `CONTENT_LENGTH`: it represents the content size of the HTTP header. <br>
# `WHOIS_COUNTRY`: it is a categorical variable, its values are the countries we got from the server response (specifically, our script used the API of Whois). <br>
# `WHOIS_STATEPRO`: it is a categorical variable, its values are the states we got from the server response (specifically, our script used the API of Whois). <br>
# `WHOIS_REGDATE`: Whois provides the server registration date, so, this variable has date values with format DD/MM/YYY HH:MM <br>
# `WHOISUPDATEDDATE`: Through the Whois we got the last update date from the server analyzed <br>
# `TCPCONVERSATIONEXCHANGE`: This variable is the number of TCP packets exchanged between the server and our honeypot client <br>
# `DISTREMOTETCP_PORT`: it is the number of the ports detected and different to TCP <br>
# `REMOTE_IPS`: this variable has the total number of IPs connected to the honeypot <br>
# `APP_BYTES`: this is the number of bytes transfered <br>
# `SOURCEAPPPACKETS`: packets sent from the honeypot to the server <br>
# `REMOTEAPPPACKETS`: packets received from the server <br>
# `APP_PACKETS`: this is the total number of IP packets generated during the communication between the honeypot and the server <br>
# `DNSQUERYTIMES`: this is the number of DNS packets generated during the communication between the honeypot and the server <br>
# `TYPE`: this is a categorical variable, its values represent the type of web page analyzed, specifically, 1 is for malicious websites and 0 is for benign websites <br>

# ## Approach:
# 1) Explore the data <br>
# 2) Clean the relevant data <br>
# 3) Check imbalances (if any) <br>
# 4) Use resampling technqiues to resolve imbalances <br>
# 5) Feature Engineering and Feature Selection <br>
# 6) Try and finalize a Machine Learning Model <br>
# 7) Validate our results on Cross-Validation set <br>
# 8) Final Inferences and Conclusion <br>

# #### Importing the required packages and dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("train.csv") # importing the dataset


# In[3]:


df.head() # checking the head


# In[4]:


df.info() # checking the info


# In[5]:


df.isna().sum() # Checking the missing values per column


# #### Checking the imbalance in target variable

# In[6]:


100 * df['Type'].value_counts()/len(df)


# There is a high imbalance towards Type=0 class. The ratio is around 88%-12%. We'll fix this later using SMOTE.

# ## Exploratory Data Analysis (EDA)

# In[7]:


df.describe() # checking the descriptive stats for each numerical feature


# ### Now let's try finding any trend of other features with respect to Target variables

# ### 1) Type vs Number of special characters

# In[8]:


plt.figure(figsize=(12,6))
sns.boxplot(data=df,x='Type',y='NUMBER_SPECIAL_CHARACTERS');


# ### Inference: 
# The number of special characters for Malicious website is higher than benign ones.

# ### 2) Type vs URL length

# In[9]:


plt.figure(figsize=(12,6))
sns.boxplot(data=df,x='Type',y='URL_LENGTH');


# #### Inference:
# The mean length of URL is higher for Malicious website as compared to benign ones.

# In[ ]:





# ## Let's do a bit of Data processing and then continue with EDA

# #### Let's see how many unique categories are there in each categorical column

# In[10]:


for i in df.select_dtypes(include='object').columns:
    print(f"{i} -> {df[i].nunique()}")


# #### Let's create some custom functions to only keep top 5 category in each categorical column and then applying these function to their respective columns

# In[11]:


df['CHARSET'].value_counts()


# In[12]:


def CHARSET_CLEANER(x):
    if x not in ['UTF-8','ISO-8859-1','utf-8','us-ascii','iso-8859-1']:
        return "OTHERS"
    else:
        return x


# In[13]:


df['CHARSET'] = df['CHARSET'].apply(CHARSET_CLEANER)


# In[14]:


df['CHARSET'].value_counts()


# In[15]:


df['SERVER'].value_counts()


# In[16]:


def SERVER_CLEANER(x):
    if x not in ['Apache','nginx','None','Microsoft-HTTPAPI/2.0','cloudflare-nginx']:
        return "OTHERS"
    else:
        return x


# In[17]:


df['SERVER'] = df['SERVER'].apply(SERVER_CLEANER)


# In[18]:


df['SERVER'].value_counts()


# In[19]:


df['WHOIS_STATEPRO'].value_counts()[:7]


# In[20]:


def STATE_CLEANER(x):
    if x not in ['CA','None','NY','WA','Barcelona','FL']:
        return "OTHERS"
    else:
        return x


# In[21]:


df['WHOIS_STATEPRO'] = df['WHOIS_STATEPRO'].apply(STATE_CLEANER)


# In[22]:


df['WHOIS_STATEPRO'].value_counts()


# In[23]:


def DATE_CLEANER(x):
    if x == 'None':
        return "Absent"
    else:
        return "Present"


# In[24]:


df['WHOIS_REGDATE'] = df['WHOIS_REGDATE'].apply(DATE_CLEANER)


# In[25]:


df['WHOIS_UPDATED_DATE'] = df['WHOIS_UPDATED_DATE'].apply(DATE_CLEANER)


# In[26]:


df.head()


# In[27]:


df.drop(['URL','WHOIS_COUNTRY'],axis=1,inplace=True)


# In[28]:


df.head()


# ## EDA continued

# ### 3) Correlation Heat map

# In[29]:


plt.figure(figsize=(20,10))
sns.heatmap(data=df.corr(),cmap='plasma',annot=True)


# #### Although we can see some highly correlated features, it won't be wise to remove them all as that could lead us to significant loss in drawing out inferences. Hence, we'll only remove those columns which won;t have much impact on analysis and frther modelling

# #### Since Content Length is not significantly correlated with any of the features and also contains a lot of missing values. It would be good if we drop it out.

# In[30]:


df2 = df.copy() # creating a copy of our dataframe


# In[31]:


df2.drop("CONTENT_LENGTH",axis=1,inplace=True) # dropping the column which is not required


# ## Feature Engineering and Feature Selection

# #### Changing categorical column into dummies

# In[32]:


df3 = df2.copy() # creating a copy of the dataframe


# In[33]:


df3 = pd.get_dummies(df3,columns=['WHOIS_UPDATED_DATE','WHOIS_REGDATE','WHOIS_STATEPRO','SERVER','CHARSET'],drop_first=True) # creating dummies


# In[34]:


df3.head() # checking the head


# In[35]:


df3.isna().sum() # checking for any missing value


# In[36]:


df3.dropna(inplace=True) # dropping all the missing values


# ### Using SMOTE to extrapolate our model

# `About SMOTE`: Synthetic Minority Oversampling Technique (SMOTE) is a resampling technique which oversamples the minority class by "synthesizing" various parameters and creating new data points by using various "data augmentation" techniques. By this, we get enough numbers of minority class data points to sufficiently carry out the learning processes for the ML model. Hence, we avoid data duplication of minority class (which is the case in oversampling).  

# In[37]:


# Importing the SMOTE function
from imblearn.over_sampling import SMOTE


# In[38]:


# Creating the set of independent features and target variable
X = df3.drop("Type",axis=1)
y = df3['Type']


# In[39]:


from imblearn.under_sampling import RandomUnderSampler  # importing the Under Sampling function


# In[40]:


# We shall keep undersampled majority class 50% more than the oversampled minority class. 
# This is being done on order to resemble the composition of original dataframe in the SMOTE's dataframe
undersample = RandomUnderSampler(sampling_strategy=0.5) 


# In[41]:


from imblearn.pipeline import Pipeline # Importing the pipeline


# In[42]:


# Initializing the SMOTE function. We set our SMOTE function to oversample the minority to the number equal to the majority class. 
#Then, we take 50% of the oversampled minority class (randomly sampled).
oversample = SMOTE(sampling_strategy=0.5) 


# In[43]:


steps = [('o',oversample),('u',undersample)] # steps for pipelining. First "do oversampling of the minority class" and then do "undersampling of the majority class"


# In[44]:


pipeline = Pipeline(steps=steps) # Creating the pipeline instance


# In[45]:


X_smote, y_smote = pipeline.fit_resample(X,y) # Fitting the pipeline to our dataset


# In[46]:


y_smote.value_counts() # Taking value counts of the targte feature


# In[47]:


len(X_smote) # checking the total number of samples we have


# In[48]:


X_smote.shape # checking the shape


# ### SMOTE has been implemented. Now, we'll start the modelling by first creating a hold-out train and test set and then using stratified cross validation to cover all possibilities

# In[49]:


from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,cross_validate  # Implementing the required functions


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # creating a test holdout set


# In[51]:


from sklearn.preprocessing import StandardScaler # import the standard scaling function


# In[52]:


sc = StandardScaler() # creating an instance of the scaling function


# In[53]:


X_train = sc.fit_transform(X_train) # fitting and transform the training set
X_test = sc.transform(X_test) # just transforming the testing set to avoid 'data leakage'


# In[54]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,fbeta_score,make_scorer,precision_score,recall_score 
# importing all the metric scores required for evaluation


# In[55]:


# creating a dictionary to evaluate metric over stratified k-fold cv
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}


# In[ ]:





# ### We'll be using tree based ensemble models as they are immune to multicollinearity

# ### Note: We're interested in a model which has high recall as we want to minimize False Negative Rate at the same time keeping precision high as well. Hence, we need to find a `sweet spot` while evaluating our model

# ### 1) Random Forest

# In[56]:


from sklearn.ensemble import RandomForestClassifier # importing the function


# In[57]:


rf = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=42,class_weight={0:1,1:5},max_depth=5) # creating an instance


# In[58]:


rf.fit(X_train,y_train) # fitting the model


# In[59]:


rf_cv_f1 = cross_validate(rf,X_test,y_test,cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=111),scoring=scoring) 
# cross-validating our model over 5 folds and evaluting metrics are: accuracy, precision, recall and F-1 score


# In[60]:


print(f" ACCURACY: {rf_cv_f1['test_accuracy'].mean()}")
print(f" PRECISION: {rf_cv_f1['test_precision'].mean()}")
print(f" RECALL: {rf_cv_f1['test_recall'].mean()}")
print(f" F-1 Score: {rf_cv_f1['test_f1_score'].mean()}")


# In[61]:


rf_pred = rf.predict(X_test) # predicting on the hold out test set


# In[62]:


print(classification_report(y_test,rf_pred)) 
print(confusion_matrix(y_test,rf_pred))


# In[ ]:





# ### 2) Catboost

# In[63]:


from catboost import CatBoostClassifier # importing the function


# In[64]:


cb = CatBoostClassifier(random_state=42,verbose=500,class_weights={0:1,1:5},max_depth=5,early_stopping_rounds=30,boosting_type='Ordered') # creating an instance


# In[65]:


cb.fit(X_train,y_train) # fitting the model


# In[66]:


cb_cv_f1 = cross_validate(cb,X_test,y_test,cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42),scoring=scoring)
# cross-validating our model over 5 folds and evaluting metrics are: accuracy, precision, recall and F-1 score


# In[67]:


print(f" ACCURACY: {cb_cv_f1['test_accuracy'].mean()}")
print(f" PRECISION: {cb_cv_f1['test_precision'].mean()}")
print(f" RECALL: {cb_cv_f1['test_recall'].mean()}")
print(f" F-1 Score: {cb_cv_f1['test_f1_score'].mean()}")


# In[68]:


cb_pred = cb.predict(X_test) # predicting on the hold out test set


# In[69]:


print(classification_report(y_test,cb_pred))
print(confusion_matrix(y_test,cb_pred))


# In[ ]:





# ### Catboost Classifier is the best optimal model for our data as it is quite robust, immune to multicollinearity and has high recall and F-1 score

# ### Knowing the dependence of target variable on various feature using Mutual Information Gain 

# #### Mutual Information Gain:
# MI Estimate mutual information for a discrete target variable.
# 
# Mutual information (MI) between two random variables is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency. <br>
# 
# I(X ; Y) = H(X) – H(X | Y) Where I(X ; Y) is the mutual information for X and Y, H(X) is the entropy for X and H(X | Y) is the conditional entropy for X given Y. The result has the units of bits.
# 
# Credits: Krish Naik GitHub repository (Feature Selection) <br>
# Link: https://github.com/krishnaik06/Complete-Feature-Selection/blob/master/3-%20Information%20gain%20-%20mutual%20information%20In%20Classification.ipynb

# #### Before evaluating Mutual information gain, first let's remove those columns which are highly correlated, i.e. we'll first remove high multicollinearity.

# In[70]:


def correlation(dataset,threshold):
    col_corr = set() # empty set to avoid repittion later
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i,j]) > threshold: # abs is taken to consider highly negatively correlated columns as well
                colname = corr_matrix.columns[i] # getting the name of the column
                col_corr.add(colname)
    return col_corr


# In[71]:


correlation(X_smote,0.7) # all those columns which ahve more than 70% collinearity


# In[72]:


X_smote2 = X_smote.drop(list(correlation(X_smote,0.7)),axis=1) # removing all those columns which ahve more than 70% collinearity


# In[74]:


from sklearn.feature_selection import mutual_info_classif
# determine the mutual information
mutual_info = mutual_info_classif(X_smote2, y_smote)
mutual_info


# In[75]:


mutual_info = pd.Series(mutual_info) # Creating series of column names and their respective mutual information gain
mutual_info.index = X_smote2.columns # setting up index
mutual_info.sort_values(ascending=False) # sorting the values


# In[81]:


# Bar Plot of Mutual Information Gain with respect to our target variable
plt.ylabel("Mutual Information Gain")
plt.xlabel("Independent Features")
plt.title("Mutual Information Gain of each feature with respect to target variable")
mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 10));


# ### Features like URL_LENGTH, DIST_REMOTE_TCP_PORT, presence of an UPDATE_DATE, DNS_QUERY_TIMES , operation from California state etc. are among the most important features for predicting whether a website is malicious or not

# In[ ]:




