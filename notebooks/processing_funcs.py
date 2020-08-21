import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import iplot, plot
import plotly.express as px

import glob
import os

plt.rcParams["figure.figsize"] = 16,10



def data_processing(df, sample_rate):

    #creating time groups in order to evenly sample the time points
    df["time_groups"] = (df["time"] / sample_rate).apply(lambda x: int(x))

    #dividing the time groups by 1/sample rate --> correct time stamps
    df = df.groupby("time_groups").mean()
    df["time"] = df.index/(1/sample_rate)


    time = df["time"]

    gps_data = df[['lat', 'lon', 'height', 'velocity', 'direction', 'h_accuracy',
                   'v_accuracy']]
    gps_data = pd.concat([gps_data,
                          pd.DataFrame(columns=["helper_1"],
                                       data = np.where(gps_data["lat"] >0,1,0))],
                         axis=1)
    gps_data["id_gps"] = gps_data["helper_1"].cumsum()

    gps_data.fillna(method="ffill", inplace = True)
    gps_data.drop("helper_1", axis = 1, inplace = True)


    sensor_data = df[['x_lin_acc', 'y_lin_acc', 'z_lin_acc',
                     'x_gyro', 'y_gyro','z_gyro',
                     'x_acc', 'y_acc', 'z_acc']].interpolate(method='polynomial', order=2)

    df = pd.concat([time,gps_data,sensor_data], axis=1).dropna()

    df["time_shift"] = df["time"].shift()

    if round((df["time"] - df["time_shift"]).max(), 2) > sample_rate:
        pass

    df.drop("time_shift", axis = 1, inplace = True)

    return df

def read_measurement(path, sample_rate):

    df = pd.DataFrame()

    c = 0

    for path in glob.glob(path):

        new_df = pd.read_csv(path)
        new_df["sensor"] = path.split('.csv')[0].split("\\")[1]
        df = pd.concat([new_df,df], axis=0)
        c+=1

    df = df.pivot_table(index="Time (s)", columns="sensor")

    df = df.reset_index().sort_values("Time (s)")

    new_columns = [
                    "time", "direction", "height", "h_accuracy", "lat", "lon", "velocity", "v_accuracy",
                    "x_acc", "x_lin_acc", "x_gyro",
                    "y_acc", "y_lin_acc", "y_gyro",
                    "z_acc", "z_lin_acc", "z_gyro"
                ]
    df.columns = [i for i in new_columns]

    df =df[["time","lat", "lon", "height",
            "velocity", "direction",
            "h_accuracy", "v_accuracy",
            "x_lin_acc", "y_lin_acc", "z_lin_acc",
            "x_gyro", "y_gyro", "z_gyro",
            "x_acc","y_acc","z_acc", ]]


    df = data_processing(df,sample_rate)

    return df
<<<<<<< HEAD
||||||| d1c0fd2
        
=======

def read_all(sample_rate = 0.01):

    q1 = read_measurement("../data/raw_data_train/rsq_q1/*",sample_rate=sample_rate)
    q1["label"] = 0
    q2 = read_measurement("../data/raw_data_train/rsq_q2/*", sample_rate=sample_rate)
    q2["label"] = 1
    q3 = read_measurement("../data/raw_data_train/rsq_q3/*", sample_rate=sample_rate)
    q3["label"] = 2


    df = pd.concat([q1,q2,q3],axis=0)

    cols = list(df.columns)
    cols.remove("time")
    return df[["time"] + cols]
>>>>>>> feature-feature_engineering
