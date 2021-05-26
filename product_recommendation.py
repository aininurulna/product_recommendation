#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#import data
header_list = ['cust', 'product', 'rating', 'timestamp']
df = pd.read_csv('ratings_Electronics.csv', names=header_list)
data = df.head(100000)


# In[3]:


#check data information
data.info()
data.nunique()


# In[4]:


#check the distribution of customers for each rating
import seaborn as sns
sns.set_theme(style="white")
ax = sns.countplot(x="rating", data=data)


# In[5]:


#roughly check the number of products rated per customer
products_per_customer = data.groupby('cust')['product'].count().sort_values(ascending=False)
products_per_customer.head(10)


# In[6]:


#drop the timestamp column
data.drop(['timestamp'], axis=1,inplace=True)


# In[7]:


#preparing the data for modeling with KNNWithMeans by Surprise
from surprise import Dataset
from surprise import Reader
import os
from surprise.model_selection import train_test_split
def prepare (data):
    reader = Reader(rating_scale=(1, 5))
    data2 = Dataset.load_from_df(data,reader)
    train, test = train_test_split(data2, train_size=0.8, random_state=123)
    return train, test
train, test = prepare(data)


# In[8]:


from surprise import KNNWithMeans
from surprise import accuracy
knnm = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})


# In[9]:


#Memory based collaborative filtering (item-item recommendation) using KNNWithMeans
from surprise import KNNWithMeans
from surprise import accuracy
knnm = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
knnm.fit(train)
predictionmemory = knnm.test(test)
rmse_memory = accuracy.rmse(predictionmemory)


# In[10]:


#check the recommendation system by selecting one customer to recommend 10 products to that customer
#a little tip, to check the recommendation system, choose the customer who purchased a lot of items by checking the products_per_customer dataframe or pick a random numbre of the index
custid = list(data['cust'].unique())
chosen_cust = custid[9343]
print("Chosen customer " + str(chosen_cust))


# In[11]:


#list the products purchased by the chosen customer
purchased = train.ur[train.to_inner_uid(chosen_cust)]

print("Chosen customer purchased the following items ")
for items in purchased[0]: 
    print(knnm.trainset.to_raw_iid(items))


# In[14]:


#create a function to return the recommended items by the recommendation system by KNNWithMeans
def recommendbyknn(purchased, chosen_cust):
    knnm_product = knnm.get_neighbors(purchased[0][0], 10) #return the 10 nearest neighbors of iid, which is the inner id of a customer or an item

    recommendedknn = []
    for products in knnm_product:
        if not products in purchased[0]:
            to_be_purchased = knnm.trainset.to_raw_iid(products)
            recommendedknn.append(to_be_purchased)
    return recommendedknn
recommendedknn = recommendbyknn(purchased, chosen_cust)
recommendedknndf = pd.DataFrame(recommendedknn, columns=['iid'])
display(recommendedknndf)


# In[15]:


#create a model-based collaborative filtering using SVD
from surprise import SVD

svd = SVD()
svd.fit(train)
predictionmodel = svd.test(test)
rmse_memory = accuracy.rmse(predictionmodel)


# In[16]:


#list the products purchased by the chosen customer
print("Chosen customer purchased the following items ")
for items in purchased[0]: 
    print(svd.trainset.to_raw_iid(items))


# In[21]:


#create a function to return the recommended items by the recommendation system by SVD
all_products = list(data['product'].unique())

def recommendbysvd(purchased, chosen_cust):
    recommendedsvd = []

    for products in all_products:
        result = svd.predict(chosen_cust,  products, r_ui=4, verbose=True)
        recommendedsvd.append(result)
    return recommendedsvd

recommendedsvd = recommendbysvd(purchased, chosen_cust)
recommendedsvddf = pd.DataFrame(recommendedsvd, columns=['uid', 'iid', 'rui', 'est', 'details'])
recommendedsvddf['err'] = abs(recommendedsvddf.est - recommendedsvddf.rui)
recommendedsvddf = recommendedsvddf.sort_values(by=['err'], ascending=False)
recommendedsvddf.head(10)


# In[ ]:




