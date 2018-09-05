# basic
from datetime import date, timedelta
import pandas as pd
import numpy as np

# helper functions for feature engineering
def get_timespan(df, dt, minus, periods):
    return df[
        pd.date_range(dt - timedelta(days=minus), periods=periods)
    ]

def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values, # mean target values for past 3, 7, and 14 days
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values # number of times item was on promo in lsat 14 days
    })
    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[ # these features indicate which of the future days the item is on promo
            t2017 + timedelta(days=i)].values.astype(np.uint8)
    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    else:
        y = df_2017[
            pd.date_range(t2017, periods=1)
        ].values
        return X, y

if __name__ == '__main__':
    
    # import data
    df_train = pd.read_csv(
        'data/raw_data/train.csv', usecols=[1, 2, 3, 4, 5],
        dtype={'onpromotion': bool},
        converters={'unit_sales': lambda u: np.log1p( # need to log transform the target variable
            float(u)) if float(u) > 0 else 0},
        parse_dates=["date"],
    )
    
    df_test = pd.read_csv(
        'data/raw_data/test.csv', usecols=[0, 1, 2, 3, 4],
        dtype={'onpromotion': bool},
        parse_dates=["date"]
    ).set_index(
        ['store_nbr', 'item_nbr', 'date']
    )
    
    items = pd.read_csv(
        'data/raw_data/items.csv',
    ).set_index("item_nbr")
    
    # only keep 2017 data
    df_2017 = df_train[df_train.date.isin(
        pd.date_range("2017-05-01", periods=7 * 14))].copy()
    del df_train
    
    # combine promotion data
    promo_2017_train = df_2017.set_index(
        ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
            level=-1).fillna(False)
    promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
    promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
    promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
    promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
    promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
    del promo_2017_test, promo_2017_train
    
    # multi-index data
    df_2017 = df_2017.set_index(
        ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
            level=-1).fillna(0)
    df_2017.columns = df_2017.columns.get_level_values(1)
    
    items = items.reindex(df_2017.index.get_level_values(1))
    
    print("Preparing dataset...")
    t2017 = date(2017, 6, 5)
    X_l, y_l = [], []
    for i in range(4):
        delta = timedelta(days=7 * i)
        X_tmp, y_tmp = prepare_dataset(
            t2017 + delta
        )
        print(t2017+delta)
        X_l.append(X_tmp)
        y_l.append(y_tmp)
    X_train = pd.concat(X_l, axis=0)
    y_train = np.concatenate(y_l, axis=0)
    del X_l, y_l
    X_val, y_val = prepare_dataset(date(2017, 7, 10))
    X_test, y_test = prepare_dataset(date(2017, 7, 31), is_train=False)
    
    # output clean data
    X_train.to_csv('X_train.csv', index=False)
    X_val.to_csv('X_val.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    
    y_train.to_csv('X_train.csv', index=False)
    y_val.to_csv('X_train.csv', index=False)
    y_test.to_csv('X_train.csv', index=False)