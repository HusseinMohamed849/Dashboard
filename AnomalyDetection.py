#Data manipulation modules
import pandas as pd

#Data visualization modules
import plotly.graph_objs as go
import plotly.express as px

#Model and Components_plot
from fbprophet import Prophet
from fbprophet.plot import plot_components_plotly

#Error metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.metrics import r2_score, median_absolute_error

from datetime import timedelta
class OnlineDetector(object):
    def __init__(self, name, df):
        self.name = name
        self.period = 30
        self.freq = 'D'
        self.update_model(df, ['D', 'W'])
        self.metrics = {
            "MAE":mean_absolute_error,
            "R2":r2_score, 
            "MSE":mean_squared_error,
            "MSLE":mean_squared_log_error,
            "MedAE":median_absolute_error
        }
        
        
    def update_model(self, df, params):       
        self.model_init_date = df.ds.iloc[-1]

        daily = 'D' in params 
        weekly = 'W' in params
        yearly = 'Y' in params

        if daily or weekly or yearly:
            self.model = Prophet(daily_seasonality=daily, weekly_seasonality=weekly, yearly_seasonality=yearly)
        else:
            self.model = Prophet()

        self.model.fit(df)
        self.update_future_df(period=self.period, freq=self.freq)
        self.update_predict_df(series=df)

            
    def update_future_df(self, period, freq):
        self.period = period
        self.freq = freq
        data = self.model.make_future_dataframe(periods=self.period, freq=freq, include_history=True)
        self.forecast_dataframe = self.model.predict(df=data)
        self.set_forcast_plots()

    def update_predict_df(self, series):
        
        #Get predictions and classify anomalies
        self.predict_dataframe = self.model.predict(series[["ds"]])
        self.predict_dataframe['y'] = series['y']
        self.predict_dataframe["residuals"] = self.predict_dataframe['y'] - self.predict_dataframe["yhat"]
        self.classify_anomaly()

    def classify_anomaly(self, stds=[2, 4, 8]):
        # Populate errors
        error = self.predict_dataframe.residuals.abs()
        mean_of_errors = error.values.mean()
        std_of_errors = error.values.std(ddof=0)

        # initialize the anomaly data with False and impact 0
        self.predict_dataframe["anomaly"] = False
        self.predict_dataframe["impact"] = 0

        for i in range(len(stds)):
            num_stds = stds[i]
            # Define outliers by distance from mean of errors
            threshold = mean_of_errors + (std_of_errors * num_stds)
            # Label outliers using standard deviations from the errors" mean
            self.predict_dataframe.at[error > threshold, "anomaly"] = True
            self.predict_dataframe.at[error > threshold, "impact"] = i+1

    def set_forcast_plots(self):
        self.predicted_line = go.Scatter(
            x=self.forecast_dataframe.ds, 
            y=self.forecast_dataframe.yhat, 
            mode="lines", 
            name="Prediction", 
            line_color="RoyalBlue"
        )
        
        self.confidence_area = go.Scatter(
            x=pd.concat([self.forecast_dataframe.ds, self.forecast_dataframe.ds[::-1]]),
            y=pd.concat([self.forecast_dataframe.yhat_upper, self.forecast_dataframe.yhat_lower[::-1]]),
            fill="toself",
            mode="none",
            fillcolor="SteelBlue",
            opacity=0.4,
            name="confidence interval"
        )
    
    def get_anomaly_plots(self):
        plots = []

        # Normal data-points
        noraml = self.predict_dataframe[self.predict_dataframe.impact == 0]
        scatter_noraml = go.Scatter(
            x=noraml.ds, 
            y=noraml.y, 
            mode="markers", 
            name="Actual-Noraml", 
            marker_size=4,
            marker=dict(color="#CCCCFF")
        )
        plots.append(scatter_noraml)


        # Anomaly data-points
        colors = ["#F4D03F", "#F39C12", "#CB4335"]
        for i in range(1, 4):
            anomalies = self.predict_dataframe[self.predict_dataframe.impact==i]
            scatter_anomaly = go.Scatter(
                x=anomalies.ds, 
                y=anomalies.y, 
                name=f"Actual-Anomaly-Impact {i}", 
                mode="markers", 
                marker=dict(color=colors[i-1], size=4)
            )
            plots.append(scatter_anomaly)
            
        return plots
    
    def get_hist_figure(self):
        fig = px.histogram(data_frame=self.predict_dataframe, x='y', marginal="box")#,nbins=nbins)

        layout = go.Layout(
            template="plotly_dark",
            margin={'r':10, 'l':10, 'b':10},
            title = {
                "text":"<b>Histogram<b>",
                "font_size":20,
            },
            xaxis = {
                "title":self.name
            },
            yaxis = {
                "title":"Number of Samples"
            }
        )

        fig.update_traces(
            marker=dict(
                color="RoyalBlue", 
                opacity=0.4, 
                line=dict(color="#9370db", width=1.3)
            )
        )
        fig.update_layout(layout)
        return fig
    
    
    def get_components_figure(self):
        fig = plot_components_plotly(self.model, self.predict_dataframe)
        fig.update_layout(
            template="plotly_dark", 
            height=None, 
            width=None, 
            margin={'t':10, 'b':10, 'r':10, 'l':10}
        )
        fig.update_traces(line=dict(color="RoyalBlue"))
        return fig
    
    
    def get_streaming_figure(self):
        plots = [self.confidence_area, self.predicted_line, *self.get_anomaly_plots()]
        # minimumX, maximumX = min(self.forecast_dataframe.ds), max(self.forecast_dataframe.ds)
        # minimumY, maximumY = min(self.predict_dataframe.y), max(self.predict_dataframe.y)
        return {
            "data":plots,
            "layout":go.Layout(
                hovermode = 'x',
                template='plotly_dark',
                #showlegend = False,
                autosize=True,
                title = {
                    "text":"<b>Streaming<b>",
                    "font_size":20,
                },
                xaxis = {
                    # "range":[minimumX - timedelta(5), maximumX + timedelta(5)],
                    "title":"<b>Time</b>",
                    "rangeselector":dict(
                        buttons=list([
                            dict(count=1, label="1D", step="day", stepmode="backward"),
                            dict(count=6, label="1M", step="month", stepmode="backward"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(step="all")
                        ]),
                        font=dict(color="black")
                    )
                },
                yaxis = {
                    # "range":[minimumY - 2, maximumY + 2],
                    "title":f"<b>{self.name}</b>"
                },
                legend = {
                    "orientation":'h',
                    #'bgcolor':'#010915',
                    "xanchor":"center", 
                    'x':0.5, 
                    'y':-0.3
                },
            )
        }
    
    error = lambda self, metric:metric(self.predict_dataframe.y, self.predict_dataframe.yhat)
    def get_error_figure(self):
        metrics_results = {name:self.error(method) for name, method in self.metrics.items()}
        
        error_df = pd.DataFrame(metrics_results, index=[0])
        error_df = error_df.melt(var_name="Metric", value_name="Error")
        error_df.Error = error_df.Error.round(3)
        
        fig = px.bar(data_frame=error_df, x="Metric", y="Error", color="Metric", text="Error")
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(
            title = {
                "text":"<b>Error Metric Results<b>",
                "font_size":20,
            },
            template='plotly_dark', 
            margin={'r':10, 'l':10, 'b':10}, 
        )
        return fig

    # def get_residuals_figure(self):
    #     residuals_plot = go.Scatter(
    #         x=self.predict_dataframe.ds,
    #         y=self.predict_dataframe.residuals,
    #     )
    #     fig = {
    #         "data":[residuals_plot], 
    #         "layout":go.Layout(template="plotly_dark", height=None, width=None)
    #     }
    #     return fig
