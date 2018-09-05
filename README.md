# Favorita dataset modelling with PySpark MLlib


This repository contains some scripts used to make predictions on the Corporacion Favorita Grocery Sales Forecasting dataset. I did this small project in order to get more familiar with the PySpark MLlib package.

#### Data
The data contains daily sales histories of various grocery products in various stores, as well as their promotion histories (i.e. when these items were on sale). In addition, various information about the stores themselves, such as location, are provided. The data also contains the future promotion schedule for each item.

The dataset can be found here: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data

I haven't included any of the data in this repo, because they're quite big.

#### Prediction Objective
The task is to predict the daily sales volume for each item-store combination. The actual kaggle competition wanted sales predictions for the next 16 days after the dataset ended. However, since this was a learning exercise, I only predicted sales for the next day.  

#### Features
The features that I used were quite limited, as the focus of this project was on modelling rather than feature engineering. Features included:
* 3,7, and 14 day moving averages for sales past sales volume
* The number of days within the past 2 weeks that an item was on promotion
* whether or not an item is schedule to be on promotion, for each of the next 16 days

Since feature engineering was not the focus, the feature engineering in these scripts pretty much come completely from Ceshine's LightGBM starter kernel, which can be found here:
https://www.kaggle.com/ceshine/lgbm-starter

#### Modelling
I used three models from PySpark MLlib, all with default hyperparameters:
* Linear regression
* Random Forest
* Gradient Boosted Trees

I also compared them to a LightGBM model trained in-memory, which was included in Ceshine's kernel above. We know that the LightGBM model works for sure since it's just copy-pasted.

#### Model Evaluation
The training dataset spanned from June 5, 2017 - July 9, 2018 (5 weeks). The validation set spans from July 10 - July 16 (1 week). The test set is the single day of July 17. Note that since I didn't tune any hyperparameters for the PySpark models, a validations set technically isn't necessary. Therefore, for those models, I simply trained the model on both the train and val sets. The validations set was used for LightGBM for early stopping.

I used RMSE as the error metric for each of the models. Note that the target variable has been log transformed.

Model                           | RMSE                  |
------------------------------- |  -------------------: |
PySpark Linear Regression       | 0.614086              |
PySpark Random Forest           | 0.628987              |
PySpark Gradient Boosted Trees  | 0.621411              |
In-Memory LightGBM              | 0.604507              |

#### Conclusion
As expected, LightGBM performed the best. However, our other models came pretty close, which means that I probably didn't make and huge errors in the modelling process, which means that this project achieved its goal.
