#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
import pickle
nltk.download('punkt')
nltk.download('wordnet')


# In[2]:



# load data from database
engine = create_engine('sqlite:///disaster_response')
df = pd.read_sql('SELECT * FROM disaster_response', engine)
X = df.filter(items=['id', 'message', 'original', 'genre'])
y = df.drop(['id', 'message', 'original', 'genre', 'child_alone'], axis=1)#'child_alone' has no responses
#Mapping the '2' values in 'related' to '1' - because I consider them as a response (that is, '1')
y['related']=y['related'].map(lambda x: 1 if x == 2 else x)
df.head()


# ### 2. Write a tokenization function to process your text data

# In[3]:


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w).lower().strip() for w in tokens]


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[4]:



pipeline = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier())
                     ])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline.fit(X_train['message'], y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[ ]:



y_pred_test = pipeline.predict(X_test['message'])
y_pred_train = pipeline.predict(X_train['message'])
print(classification_report(y_test.values, y_pred_test, target_names=y.columns.values))
print('-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-')
print('\n',classification_report(y_train.values, y_pred_train, target_names=y.columns.values))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[ ]:


parameters = {'clf__max_depth': [10, 20, None],
              'clf__min_samples_leaf': [1, 2, 4],
              'clf__min_samples_split': [2, 5, 10],
              'clf__n_estimators': [10, 20, 40]}

cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=1, n_jobs=-1)
cv.fit(X_train['message'], y_train)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:


y_pred_test = cv.predict(X_test['message'])
y_pred_train = cv.predict(X_train['message'])
print(classification_report(y_test.values, y_pred_test, target_names=y.columns.values))
print('-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-')
print('\n',classification_report(y_train.values, y_pred_train, target_names=y.columns.values))


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:


cv.best_params_


# ### 9. Export your model as a pickle file

# In[ ]:


m = pickle.dumps('clf')


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




