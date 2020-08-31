import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import numpy as np
import dash_html_components as html
import pandas as pd
from scipy.stats import norm, kurtosis
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import plotly.express as px
from joblib import load
from zipfile import ZipFile
import base64
import datetime
import io


def data_processing(df, sample_rate):

    #creating time groups in order to evenly sample the time points
    df["time_groups"] = (df["time"] / sample_rate).apply(lambda x: int(x))

    #dividing the time groups by 1/sample rate --> correct time stamps
    df = df.groupby(["time_groups"]).mean()
    df["time"] = df.index/(1/sample_rate)

#     measurementID = df["measurementID"]

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

#     df = pd.concat([measurementID, time,gps_data,sensor_data], axis=1).dropna()
    df = pd.concat([time,gps_data,sensor_data], axis=1).dropna()

    df["time_shift"] = df["time"].shift()

    if round((df["time"] - df["time_shift"]).max(), 2) > sample_rate:
        pass

    df.drop("time_shift", axis = 1, inplace = True)

    return df

def read_measurement(archive, sample_rate):

#     archive = zipfile.ZipFile(data, "r")


    df = pd.DataFrame()

    for c in archive.filelist:

        new_df = pd.read_csv(archive.open(c))
        new_df["sensor"] = c.filename.split(".")[0]


        df = pd.concat([new_df,df], axis=0)


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

#     df["measurementID"] = int(c.split(".csv")[0].split('\\')[-2])

    df = data_processing(df, sample_rate)

    return df

def kurtosis_time(x):

    return kurtosis(x, fisher=True)

def rms_100(x):

    return np.sqrt(np.mean(x**2))

def crest(x):

    return max(abs(x))/np.sqrt(np.mean(x**2))

def create_aggregated(df):
    """Create a aggregated dataframe in time domain"""
    signals = ['x_lin_acc', 'y_lin_acc', "z_lin_acc",
               'x_acc', 'y_acc', 'z_acc',
               'x_gyro', 'y_gyro', 'z_gyro']

    agg_df = df.groupby(["id_gps"]).agg({x: ["sum", "mean", "mad",
                                                "median", "min", "max",
                                                "std", "var", "sem",
                                                "skew", "quantile",
                                                kurtosis_time, rms_100,
                                                crest] for x in signals})

    new_cols = []

    for k,i in enumerate(agg_df.columns):

        new_cols.append(i[0] + "_" +  i[1])

    agg_df.columns = new_cols

    return agg_df.reset_index()

def load_model(path):

    model = load("rfc_vfinal.joblib")

    return model

def classify_data(df, feature_df, model):

    pred = model.predict(feature_df.drop(["id_gps"], axis = 1))

    map_df = pd.concat([df[["id_gps", "lat", "lon"]]\
                    .groupby(["id_gps"])\
                    .max()\
                    .reset_index(),
                    pd.DataFrame({"label" : pred})], axis=1)

    return map_df




description = dbc.CardBody(["I live in Hungary, where the road surface quality is one of the worst in Europe. While I was on the road, I thought several times that it would be good to plan my trip depending on the road quality. In this case I could better enjoy the beauties of my country, and the trips would been more safe. Because of this, I decided to create a proof of concept in this topic.The goal of the proof of concept is to evaluate specific road surface measurements, and plot the road surface quality to a geographical map, only using the measurement data.The measurements have been recorded via a smart phone, and  only the accelerometer, gyroscope and gps data have been used, to classify the road surface into a quality class."])


upload = dbc.CardBody([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
])


col1 = dbc.Card([
    html.H1("Project Description"),
    html.P(description),
    html.Hr(),
    upload
]
)


def basic_plot():

    fig = go.Figure()
    fig = px.scatter_mapbox(
                            lat=[47.68333],
                            lon=[17.63512],
                            zoom = 12)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig.layout.coloraxis.showscale = False

    return fig


map_plot = dbc.CardBody([
    html.P(id = "data-for-plot"),
    dcc.Graph(id = "map-plot", figure = basic_plot()), ]
)

col2 = map_plot


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
application = app.server

app.layout = dbc.Container(
    [
        html.H1("Classify Road Surface Data"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(col1, md=4),
                dbc.Col(col2, md=8),
            ],
            align="center",
        ),
    ],
    fluid=True,
)

@app.callback(
    Output("map-plot", "figure"),
  [Input('upload-data', 'contents')],
  [State('upload-data', 'filename'),
   State('upload-data', 'last_modified')])

def update_output(list_of_contents, list_of_names, list_of_dates):

    for content, name, date in zip(list_of_contents, list_of_names, list_of_dates):
        # the content needs to be split. It contains the type and the real content
        content_type, content_string = content.split(',')
        # Decode the base64 string
        content_decoded = base64.b64decode(content_string)
        # Use BytesIO to handle the decoded content
        zip_str = io.BytesIO(content_decoded)
        # Now you can use ZipFile to take the BytesIO output
        zip_obj = ZipFile(zip_str, 'r')

        df = read_measurement(zip_obj,0.02)

        feature_df = create_aggregated(df)

        model = load_model("rfc_vfinal.joblib")

        result = classify_data(df, feature_df,model)


        fig = go.Figure()
        fig = px.scatter_mapbox(result,
                                lat="lat",
                                lon="lon",
                                zoom = 12,
                                #height=300,
                                color = "label",
                                color_continuous_scale=["green","red"])
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.layout.coloraxis.showscale = False

        return fig


if __name__ == '__main__':
    application.run(port = 8080)
