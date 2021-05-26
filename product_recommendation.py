#Product Recommendation with KNNWithMeans and SVD

import pandas as pd
import numpy as np
import seaborn as sns
from surprise import Dataset
from surprise import Reader
import os
from surprise.model_selection import train_test_split
from surprise import KNNWithMeans
from surprise import accuracy
from surprise import SVD

#import data
header_list = ['cust', 'product', 'rating', 'timestamp']
data = pd.read_csv('ratings_Electronics.csv', names=header_list) #or choose top 100000 if it's too big

#check data information
data.info()
data.nunique()

#check the distribution of customers for each rating
import seaborn as sns
sns.set_theme(style="white")
ax = sns.countplot(x="rating", data=data)

#roughly check the number of products rated per customer
products_per_customer = data.groupby('cust')['product'].count().sort_values(ascending=False)
products_per_customer.head(10)

#drop the timestamp column
data.drop(['timestamp'], axis=1,inplace=True)

#preparing the data for modeling with Surprise
def prepare (data):
    reader = Reader(rating_scale=(1, 5))
    data2 = Dataset.load_from_df(data,reader)
    train, test = train_test_split(data2, train_size=0.8, random_state=123)
    return train, test
train, test = prepare(data)



#Memory based collaborative filtering (item-item recommendation) using KNNWithMeans
knnm = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
knnm.fit(train)
predictionmemory = knnm.test(test)
rmse_memory = accuracy.rmse(predictionmemory)

#check the recommendation system by selecting one customer to recommend 10 products to that customer
#a little tip, to check the recommendation system, choose the customer who purchased a lot of items by checking the products_per_customer dataframe or pick a random numbre of the index
custid = list(data['cust'].unique())
chosen_cust = custid[9343]
print("Chosen customer " + str(chosen_cust))

#list the products purchased by the chosen customer
purchased = train.ur[train.to_inner_uid(chosen_cust)]

print("Chosen customer purchased the following items ")
for items in purchased[0]: 
    print(knnm.trainset.to_raw_iid(items))

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



#create a model-based collaborative filtering using SVD
svd = SVD()
svd.fit(train)
predictionmodel = svd.test(test)
rmse_memory = accuracy.rmse(predictionmodel)

#list the products purchased by the chosen customer
print("Chosen customer purchased the following items ")
for items in purchased[0]: 
    print(svd.trainset.to_raw_iid(items))

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



