#!/usr/bin/env python
# coding: utf-8

# In[180]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn import metrics


# In[181]:


shopping = pd.read_csv('online_shoppers_intention.csv')


# In[182]:


shopping


# In[183]:


shopping_preprocessed = shopping.copy()


# In[184]:


shopping_target = shopping['Revenue']


# In[185]:


shopping_preprocessed = shopping.drop(columns='Revenue')


# In[186]:



# encode the target variable into a numeric value
label_encoder = preprocessing.LabelEncoder()
shopping_target = label_encoder.fit_transform(shopping_target)


# In[187]:


shopping_target


# In[188]:


# encode 
encoder = preprocessing.OneHotEncoder()
encoded = pd.DataFrame(encoder.fit_transform(shopping_preprocessed[['Month', 'VisitorType', 'Weekend']]).toarray(), columns=encoder.get_feature_names(['Month', 'VisitorType', 'Weekend']))

shopping_preprocessed = shopping_preprocessed.drop(columns=['Month', 'VisitorType', 'Weekend'])
shopping_preprocessed = shopping_preprocessed.join(encoded)
shopping_preprocessed


# In[189]:


X = shopping_preprocessed


# In[190]:


y = shopping_target


# In[213]:


X.shape


# In[192]:


y.shape


# In[193]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,)


# In[194]:


X_train.shape


# In[195]:


y_train.shape


# In[196]:


from sklearn.preprocessing import MinMaxScaler


# In[197]:


min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)

X_test = min_max_scaler.fit_transform(X_test)
                                  
X_train
                                
X_test

                                       


# In[198]:


from sklearn import svm


# In[199]:


from sklearn.svm import SVC
from sklearn.svm import SVC


# In[200]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)


# In[201]:


y_pred = svclassifier.predict(X_test)


# In[202]:


from sklearn.metrics import classification_report, confusion_matrix


# In[203]:


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[204]:


from sklearn.model_selection import GridSearchCV 


# In[205]:


param_grid = {'C': [0.1, 1],  
              'gamma': [1, 0.1], 
              'kernel': ['rbf']} 


# In[206]:


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)


# In[207]:


grid.fit(X_train, y_train) 


# In[208]:


print(grid.best_params_) 


# In[209]:


print(grid.best_estimator_) 


# In[210]:


grid_predictions = grid.predict(X_test)


# In[211]:


print(classification_report(y_test, grid_predictions)) 


# In[ ]:





# In[ ]:




