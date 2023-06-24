#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import scipy as sp
import warnings
import datetime
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv('Admission_Predict.csv')


# In[4]:


data


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.value_counts()


# In[9]:


data.shape


# In[10]:


data.dtypes


# In[11]:


data.columns


# In[12]:


data.isnull().sum()


# In[13]:


data.isnull().any()


# In[ ]:





# In[14]:


data.hist(figsize=(20,14))
plt.show()


# In[15]:


sns.pairplot(data = data)


# In[16]:


data.corr()


# In[ ]:





# In[17]:


plt.figure(figsize = (12,10))

sns.heatmap(data.corr(), annot = True)


# In[18]:


sns.swarmplot( x="University Rating", y="GRE Score", data=data)


# In[19]:


sns.relplot(x="GRE Score",y="CGPA", data=data)


# In[20]:


sns.boxplot(x="University Rating", y="GRE Score", data=data)


# In[21]:


sns.regplot(x="TOEFL Score", y="CGPA", data=data,color='orange')


# In[22]:


sns.violinplot(x="University Rating", y="GRE Score", data=data)


# In[23]:


sns.lineplot(x="CGPA", y="GRE Score", data=data, color='purple')


# In[24]:


sns.jointplot(x="CGPA", y="Serial No.", data=data, color='g')


# In[25]:


sns.swarmplot(x="University Rating",y="CGPA", data =data)


# In[26]:


sns.barplot(x="University Rating",y="CGPA",data=data)


# In[27]:


sns.barplot(x="SOP",y="GRE Score", data=data)


# In[28]:


sns.scatterplot(x="CGPA",y="GRE Score", data=data, color='r')


# In[29]:


data.columns


# In[30]:


plt.style.use("ggplot")
sns.kdeplot(x="University Rating",y="CGPA", data= data, color='blue')


# In[31]:


x=data.drop('Chance of Admit ',axis=1)
y=data['Chance of Admit ']


# In[33]:



from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split


# In[36]:


#splitting training and testing data
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[37]:


print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))


# In[38]:


data.head()


# In[39]:


data=data.drop('Serial No.',axis=1)


# In[40]:


data.head()


# In[41]:


from sklearn.preprocessing import StandardScaler


# In[42]:


sc = StandardScaler()


# In[43]:


x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[44]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[46]:


lr =LinearRegression()
lr.fit(x_train, y_train)

svm = SVR()
svm.fit(x_train, y_train)

rf = RandomForestRegressor()
rf.fit(x_train,y_train)

gr = GradientBoostingRegressor()
gr.fit(x_train, y_train)


# In[48]:


y_pred1 = lr.predict(x_test)
y_pred2 = svm.predict(x_test)
y_pred3 = rf.predict(x_test)
y_pred4 = gr.predict(x_test)


# In[49]:


from sklearn import metrics


# In[50]:


score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)


# In[51]:


print(score1, score2, score3,score4)


# In[52]:


final_data = pd.DataFrame({'Models': ['LR', 'SVR', 'RF', 'GR'],
                          'R2_SCORE': [score1, score2, score3, score4]})


# In[53]:


final_data


# In[54]:


import seaborn as sns


# In[55]:


sns.barplot(final_data['Models'], final_data['R2_SCORE'])


# In[ ]:





# In[56]:


import numpy as np


# In[57]:


y_train=[1 if value>0.8 else 0 for value in y_train]
y_test=[1 if value>0.8 else 0 for value in y_test]

y_train = np.array(y_train)
y_test = np.array(y_test)


# In[58]:


from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# In[59]:


lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred1 = lr.predict(x_test)
print(accuracy_score(y_test,y_pred1))


# In[60]:


svm = svm.SVC()
svm.fit(x_train,y_train)
y_pred2 = svm.predict(x_test)
print(accuracy_score(y_test,y_pred2))


# In[63]:


knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred3 = knn.predict(x_test)
print(accuracy_score(y_test,y_pred3))


# In[65]:


rf = RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred4 = rf.predict(x_test)
print(accuracy_score(y_test,y_pred4))


# In[68]:


gr = GradientBoostingClassifier()
gr.fit(x_train, y_train)
y_pred5 = gr.predict(x_test)
print(accuracy_score(y_test,y_pred5))


# In[71]:


final_data=pd.DataFrame({'Models':['LR','SVC','KNN','RF','GBC'],
                        'ACC_SCORE':[accuracy_score(y_test,y_pred1),
                                    accuracy_score(y_test,y_pred2),
                                    accuracy_score(y_test,y_pred3),
                                    accuracy_score(y_test,y_pred4),
                                    accuracy_score(y_test,y_pred5),]})


# In[72]:


final_data


# In[73]:


import seaborn as sns


# In[74]:


sns.barplot(final_data['Models'],final_data['ACC_SCORE'])


# In[ ]:





# In[75]:


data.columns


# In[76]:


x = data.drop('Chance of Admit ',axis=1)


# In[77]:


y= data['Chance of Admit ']


# In[78]:


y = [1 if value>0.8 else 0 for value in y]


# In[79]:


y = np.array(y)


# In[80]:


x= sc.fit_transform(x)


# In[81]:


x


# In[82]:


gr = GradientBoostingClassifier()
gr.fit(x,y)


# In[83]:


import joblib 


# In[84]:


joblib.dump(gr, 'admission_model')


# In[85]:


model = joblib.load('admission_model')


# In[86]:


data.columns


# In[88]:


result = model.predict(sc.transform([[337,118,4,4.5,4.5,9.65,1]]))


# In[89]:


if result ==1:
    print("Chances of Admission is high")
else:
    print("Chances of Admission is low")


# In[ ]:




