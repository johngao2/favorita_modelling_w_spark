import pandas as pd
import numpy as np

# in-memory modelling
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

if __name__ == '__main__':

    X_train = pd.read_csv('X_train.csv')
    X_val = pd.read_csv('X_val.csv')
    X_test = pd.read_csv('X_test.csv')
    
    y_train = pd.read_csv('y_train.csv')
    y_val = pd.read_csv('y_val.csv')
    y_test = pd.read_csv('y_test.csv')
    
    items = pd.read_csv("items.csv",
                        ).set_index("item_nbr")
    
    # in-memory lightgbm baseline
    print("Training and predicting models...")
    params = {
        'num_leaves': 2**5 - 1,
        'objective': 'regression_l2',
        'max_depth': 8,
        'min_data_in_leaf': 50,
        'learning_rate': 0.05,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 1,
        'metric': 'l2',
        'num_threads': 4
    }
    
    MAX_ROUNDS = 1000
    val_pred = []
    test_pred = []
    cate_vars = []
    for i in range(2):
        print("=" * 50)
        print("Step %d" % (i+1))
        print("=" * 50)
        dtrain = lgb.Dataset(
            X_train, label=y_train[:, i],
            categorical_feature=cate_vars,
            weight=pd.concat([items["perishable"]] * 4) * 0.25 + 1
        )
        dval = lgb.Dataset(
            X_val, label=y_val[:, i], reference=dtrain,
            weight=items["perishable"] * 0.25 + 1,
            categorical_feature=cate_vars)
        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=50
        )
        print("\n".join(("%s: %.2f" % x) for x in sorted(
            zip(X_train.columns, bst.feature_importance("gain")),
            key=lambda x: x[1], reverse=True
        )))
        val_pred.append(bst.predict(
            X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
        test_pred.append(bst.predict(
            X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))
        
    print("Baseline LightGBM test rmse:", np.sqrt(mean_squared_error(
    y_test[:, 0], np.array(test_pred).transpose()[:, 0])))