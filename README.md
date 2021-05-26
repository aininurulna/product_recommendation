# product_recommendation
Product Recommendation with Surprise Python

Two different approaches were applied in this case; memory-based and model-based.
For the memory-based approach, KNNwithMeans from Surprise was implemented and resulted RMSE of 1.3420 (with top 100000 data points).
Meanwhile for the model-based approach, SVD from Surpirse was implemented and resulted RMSE of 1.2680 (with top 100000 data points).

Data:
The dataset is from:\
Amazon Reviews data (http://jmcauley.ucsd.edu/data/amazon/). This data contains the electronic products dataset.
