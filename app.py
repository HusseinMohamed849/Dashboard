# Standard Modules
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Custom Modules
from AnomalyDetection import OnlineDetector

# Read Data
data_train = pd.read_csv("./DataExample/TRW1MT(averaged).csv", parse_dates=[0])
data_train.columns = ["ds", 'y']
data_train.head()

# Create Anomaly Detector
sample = 370
updated_data = data_train.iloc[:sample, :]
detector = OnlineDetector(name="TRW1MT (C)", df=updated_data)

# Start the app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Anomaly Detection Dashboard"
app.layout = html.Section([
    # Banner:Title
    html.Div(className="banner_tsa", children=[
        html.Header(html.H1(className="titel_tsa", children=[
            "On-line", html.Span("Anomaly Detection", className="title_st"),
        ]))
    ]),

    html.Div(className="create_container", children=[
        # Options
        html.Div(className="three columns card-display", children=[
            html.P("Model Controls", className="label-control"), 
            
            html.P("Model Parameters", className="label"),
            dcc.Checklist(
                id="model_parameters_checklist",
                options=[{
                    "label":interval,
                    "value":interval[0]
                } for interval in ["Daily Seasonality", "Weekly Seasonality", "Yearly Seasonality"]],
                value=['W', 'Y'],
                className="model-parameters-checklist",
                style={"color":"white", "margin-left":"20px"}
            ),

            html.P("# of Days Forecasted", className="label"),
            dcc.Slider(
                id="forecast_days_slider",
                marks={str(days):str(days) for days in [7, 30, 90, 180, 365]},
                value=30,
                included=True,
                min=7, 
                max=365, 
                step=1, 
                updatemode="drag",
                className="forecast-days-slider"
            ),
            
            html.P("Model Updating", className="label"),
            dcc.RadioItems(
                id="updating_interval",
                options=[{"label":interval, "value":days} for interval, days in zip(
                    ["Weekly", "Monthly", "Yearly"], 
                    [7, 30, 365]
                )],
                value=7,
                style={"color":"white", "margin-left":"20px"},
                className="updating-interval"
            ),
        ]),
        
        # Streaming
        html.Div(className="ten columns card-display", children=[
            dcc.Interval(id="update_chart", interval=10000, n_intervals=sample+1),
            dcc.Graph(id="timeseries", config={"displayModeBar":"hover"}, figure=detector.get_streaming_figure()),            
        ]),
    ]),

    html.Div(className="create_container", children=[
        # Histogram
        html.Div(className="four columns card-display", children=[#animate=True
            dcc.Graph(id="histogram", config={"displayModeBar":"hover"}, figure=detector.get_hist_figure()),
        ], style={'diplay':'inline-block'}),
        # Model-components
        html.Div(className="five columns card-display", children=[
            dcc.Graph(id="seasonal_components", config={"displayModeBar":"hover"}, figure=detector.get_components_figure()),
        ]),
        # Error-barchart
        html.Div(className="four columns card-display four columns", children=[
            dcc.Graph(id="error-barchart", config={"displayModeBar":"hover"}, figure=detector.get_error_figure()),
        ]),
    ], style={"margin":"auto", "padding":"auto"}),
])


@app.callback(
    Output("timeseries", "figure"),
    Output("histogram", "figure"),
    Output("seasonal_components", "figure"),
    Output("error-barchart", "figure"),
    
    Input("update_chart", "n_intervals"),
    Input("forecast_days_slider", "value"),
    Input("model_parameters_checklist", "value"),
    Input("updating_interval", "value"),
)
def update_graphs(index, period, params, interval):
    updated_data = data_train.iloc[0:index, :]

    # Update model
    days = (updated_data.ds.iloc[-1] - detector.model_init_date).days
    if days >= interval:
        detector.update_model(updated_data, params)
        detector.model_init_date = updated_data.ds.iloc[-1]

    # Update forecasted figure
    if (detector.period != period) or (index >= len(detector.forecast_dataframe)):
        detector.update_future_df(period=period, freq='D')
                
    # Update streaming data
    detector.update_predict_df(series=updated_data)

    return [detector.get_streaming_figure(), detector.get_hist_figure(), 
            detector.get_components_figure(), detector.get_error_figure()]


if __name__=="__main__":
    app.run_server(host="127.0.0.1", port="8050", debug=False)
