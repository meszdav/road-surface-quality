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
                   'v_accuracy']].fillna(method="ffill")

    sensor_data = df[['x_lin_acc', 'y_lin_acc', 'z_lin_acc',
                     'x_gyro', 'y_gyro','z_gyro',
                     'x_acc', 'y_acc', 'z_acc']]\
                     .interpolate(method='polynomial', order=2)

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
        
