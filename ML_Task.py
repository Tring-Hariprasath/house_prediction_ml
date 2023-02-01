#!/usr/bin/env python
# coding: utf-8

# In[181]:


import pandas as pd
import matplotlib.pyplot as plt


# In[115]:


df = pd.read_csv("housing.csv")
df.head(10)


# In[116]:


df.isnull().sum()


# In[117]:


target = df['median_house_value']
target


# In[118]:


inputs = df.drop('median_house_value',axis='columns')
inputs


# In[ ]:


# converts categorical data into dummy numerical values using pandas.get_dummies() method.


# In[119]:


dummies = pd.get_dummies(df.ocean_proximity)
dummies


# In[120]:


merged = pd.concat([df,dummies],axis="columns")
merged


# In[121]:


final = merged.drop(['ocean_proximity','INLAND'],axis='columns')
final


# In[122]:


x=final.drop('median_house_value',axis='columns')
x


# In[123]:


y = df['median_income']
y


# In[159]:


df_bedrooms = final[['total_bedrooms']]
df_bedrooms


# In[160]:


df_bedrooms.isnull().sum()


# In[ ]:


#Replace missing values using SimpleImputer


# In[161]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(df_bedrooms)
X_imputed = imputer.transform(df_bedrooms)
X_imputed


# In[ ]:


#convert array into dataframe


# In[162]:


convert_to_df = pd.DataFrame(X_imputed,columns=['median_bedrooms'])
convert_to_df


# In[ ]:


#concat the above dataframe into main dataframe


# In[168]:


merged_with_final = pd.concat([final,convert_to_df],axis='columns')
merged_with_final


# In[169]:


merged_with_final = merged_with_final.drop('total_bedrooms',axis='columns')
merged_with_final


# In[ ]:


#Data points that are far from 99% percentile and less than 1 percentile are considered an outlier.


# In[176]:


df_outlier = merged_with_final['median_income']
df_outlier.shape


# In[177]:


percentile_99 = df_outlier.quantile(0.99)


# In[178]:


merged_with_final[df_outlier>percentile_99]


# In[179]:


final_outlier = merged_with_final[df_outlier<=percentile_99]
final_outlier.shape


# In[180]:


final_outlier


# In[ ]:


#split the dependent and independent variables


# In[182]:


X = final_outlier.drop(['median_house_value'],axis='columns')
X


# In[183]:


Y = final_outlier['median_house_value']
Y


# In[ ]:


#using train_test_split() method to estimate the performance of machine learning algorithms


# In[186]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2, random_state=42)
len(X_train)


# In[187]:


len(X_test)


# In[188]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train,y_train)


# In[189]:


clf.predict(X_test)


# In[191]:


y_test


# In[192]:


clf.score(X_test,y_test)


# In[194]:


final_outlier.corr()


# In[199]:


import seaborn as sns
plt.figure(figsize=(15,8))
sns.heatmap(final_outlier.corr(),annot=True,cmap="BuPu")
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[200]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X_train,y_train)


# In[203]:


X_test


# In[204]:


y_test


# In[206]:


forest.predict(X_test)


# In[207]:


forest.score(X_test,y_test)


# In[ ]:




