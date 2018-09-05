import pandas as pd

# basic pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *

# pyspark modelling
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GBTRegressor

if __name__ == '__main__':
    # read csv's
    #
    # note that normally this data would be stored as a hive table, but since those database
    # locations might be private, for the moment I'm storing them as CSV's, 
    # importing with pandas, and then converting to pyspark
    
    X_train = pd.read_csv('X_train.csv')
    X_val = pd.read_csv('X_val.csv')
    X_test = pd.read_csv('X_test.csv')
    
    y_train = pd.read_csv('y_train.csv')
    y_val = pd.read_csv('y_val.csv')
    y_test = pd.read_csv('y_test.csv')
    
    X_train['label'] = y_train[:,0]
    X_val['label'] = y_val[:,0]
    X_test['label'] = y_test[:,0]
    
    # aggregate train and val data
    train = X_train.append(X_val, ignore_index=True) # ditching val set since we aren't tuning hyperparameters
    test = X_test
    
    # combine X an y into one df, with y values given a 'label' column
    X_train = X_train.drop(['label'], axis=1)
    X_val = X_val.drop(['label'], axis=1)
    X_test = X_test.drop(['label'], axis=1)
    
    # specify pySpark df schema
    mySchema = StructType([StructField("mean_14_2017", FloatType(), True),
                           StructField("mean_3_2017", FloatType(), True),
                           StructField("mean_7_2017", FloatType(), True),
                           StructField("promo_14_2017", IntegerType(), True),
                           StructField("promo_0", IntegerType(), True),
                           StructField("promo_1", IntegerType(), True),
                           StructField("promo_2", IntegerType(), True),
                           StructField("promo_3", IntegerType(), True),
                           StructField("promo_4", IntegerType(), True),
                           StructField("promo_5", IntegerType(), True),
                           StructField("promo_6", IntegerType(), True),
                           StructField("promo_7", IntegerType(), True),
                           StructField("promo_8", IntegerType(), True),
                           StructField("promo_9", IntegerType(), True),
                           StructField("promo_10", IntegerType(), True),
                           StructField("promo_11", IntegerType(), True),
                           StructField("promo_12", IntegerType(), True),
                           StructField("promo_13", IntegerType(), True),
                           StructField("promo_14", IntegerType(), True),
                           StructField("promo_15", IntegerType(), True),
                           StructField("label", FloatType(), True)
                          ])
    
    # create spark dataframes
    train_spark = spark.createDataFrame(train, schema=mySchema)
    test_spark = spark.createDataFrame(test, schema=mySchema)
    
    # pyspark mllib requires all features to be one column vector, with each element being another vector of a row of features
    vectorAssembler = VectorAssembler(inputCols = ['mean_14_2017', 'mean_3_2017', 'mean_7_2017', 'promo_14_2017',
                                                   'promo_0', 'promo_1', 'promo_2', 'promo_3', 'promo_4',
                                                   'promo_5', 'promo_6', 'promo_7', 'promo_8', 'promo_9',
                                                  'promo_10', 'promo_11', 'promo_12', 'promo_13', 'promo_14',
                                                  'promo_15'], outputCol = 'features')
    
    train_spark = vectorAssembler.transform(train_spark)
    test_spark = vectorAssembler.transform(test_spark)
    
    train_spark = train_spark.select(['features', 'label'])
    test_spark = test_spark.select(['features', 'label'])
    
    # detects anything with less than 2 unique values as categorical
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=2).fit(train_spark)
    
    # Pyspark linear regression
    
    lr = LinearRegression(featuresCol ='features', labelCol = 'label')
    lr_pipeline = Pipeline(stages=[featureIndexer, lr])
    
    lr_model = lr_pipeline.fit(train_spark)
    lr_predictions = lr_model.transform(test_spark)
    
    lr_evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(lr_predictions)
    print("OLS test set RMSE = %g" % rmse)
    
    # random forest
    
    rf = RandomForestRegressor(featuresCol="features", labelCol='label')
    rf_pipeline = Pipeline(stages=[featureIndexer, rf])
    
    rf_model = rf_pipeline.fit(train_spark)
    rf_predictions = rf_model.transform(test_spark)
    
    rf_evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(rf_predictions)
    print("RF test set RMSE = %g" % rmse)
    
    # GBT's
    
    gbt = GBTRegressor(featuresCol = 'features', labelCol = 'label')
    gbt_pipeline = Pipeline(stages=[featureIndexer, gbt])
    
    gbt_model = gbt_pipeline.fit(train_spark)
    gbt_predictions = gbt_model.transform(test_spark)
    
    gbt_evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(gbt_predictions)
    print("GBT test set RMSE = %g" % rmse)