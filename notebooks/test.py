# # --------------------------- #
# #       Import Libraries      #
# # --------------------------- #

# import sys
# import os
# from pathlib import Path
# from itertools import cycle
# import multiprocessing

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm.notebook import tqdm

# import plotly.graph_objects as go
# import plotly.subplots as sp
# import plotly.io as pio
# import plotly.express as px

# import torch
# import pytorch_lightning as pl
# from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
# import shutil

# # Set Plotly Default Template
# pio.templates.default = "plotly_white"

# # Configure System Path
# sys.path.append(os.path.abspath('../'))

# # Import Custom Modules
# from src.utils import plotting_utils
# from src.dl.dataloaders import TimeSeriesDataModule
# from src.dl.multivariate_models import SingleStepRNNConfig, SingleStepRNNModel

# # Set Random Seed and Configure tqdm
# np.random.seed()
# tqdm.pandas()

# # Enable Inline Plotting for Matplotlib
# %matplotlib inline

# # --------------------------- #
# #   GPU Configuration Setup   #
# # --------------------------- #

# # Check GPU Availability
# gpu_available = torch.cuda.is_available()
# print(f"GPU Available: {gpu_available}")

# if gpu_available:
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
#     # Utilize Tensor Cores for Compatible GPUs
#     torch.set_float32_matmul_precision('high')  # Options: 'high', 'medium', 'default'
# else:
#     print("Using CPU")

# # --------------------------- #
# #   Data Loading & Preproc    #
# # --------------------------- #

# # Define Source Data Path
# source_data = Path("../data/")

# # Load Data
# data = pd.read_csv(source_data / "processed" / "merged_nhs_covid_data.csv")
# print("Initial Data Head:")
# display(data.head())

# # Display Unique Area Names
# unique_areas = data['areaName'].unique()
# print(f"Unique Areas: {unique_areas}")

# # Aggregate Data to 'England' Region
# data = data.groupby('date').agg({
#     'covidOccupiedMVBeds': 'sum',
#     'cumAdmissions': 'sum',
#     'hospitalCases': 'sum',
#     'newAdmissions': 'sum',
#     'new_confirmed': 'sum',
#     'new_deceased': 'sum',
#     'cumulative_confirmed': 'sum',
#     'cumulative_deceased': 'sum',
#     'population': 'sum',
#     'openstreetmap_id': 'first',
#     'latitude': 'first',
#     'longitude': 'first'
# }).reset_index()
# data['areaName'] = 'England'

# # Create Time Series Features
# data['date'] = pd.to_datetime(data['date'])
# data['year'] = data['date'].dt.year
# data['month'] = data['date'].dt.month
# data['day'] = data['date'].dt.day
# data['day_of_week'] = data['date'].dt.dayofweek

# # --------------------------- #
# #      Vax Index Function     #
# # --------------------------- #

# def calculate_vax_index(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calculate the Vax index based on vaccination rates and efficacy across age groups.
    
#     Args:
#         df (pd.DataFrame): DataFrame containing 'date' and other relevant columns.
    
#     Returns:
#         pd.DataFrame: DataFrame with an additional 'Vax_index' column.
#     """
#     # Constants
#     total_population = 60_000_000
#     number_of_age_groups = 5
#     vaccine_efficacy_first_dose = [0.89, 0.427, 0.76, 0.854, 0.75]
#     vaccine_efficacy_second_dose = [0.92, 0.86, 0.81, 0.85, 0.80]
#     age_group_probabilities_icu = [0.01, 0.02, 0.05, 0.1, 0.15]
#     monthly_vaccination_rate_increase = 0.05
#     vaccination_start_date = pd.Timestamp('2021-01-18')
    
#     # Population per Age Group
#     population_per_age_group = total_population / number_of_age_groups
    
#     # Initialize Vax Index List
#     vax_index_list = []
    
#     # Monthly Vaccination Rate (Starting from 0)
#     monthly_vaccination_rate = 0.0
    
#     for index, row in df.iterrows():
#         # Increment Monthly Vaccination Rate on the First Day of Each Month After Start Date
#         if row['date'].day == 1 and row['date'] >= vaccination_start_date:
#             monthly_vaccination_rate += monthly_vaccination_rate_increase
#             # Ensure Vaccination Rate Does Not Exceed 1
#             monthly_vaccination_rate = min(monthly_vaccination_rate, 1.0)
#             print(f"Updated monthly vaccination rate to {monthly_vaccination_rate} on {row['date'].date()}")
        
#         Si_sum = 0.0
        
#         for i in range(number_of_age_groups):
#             # Vaccinated Population for This Age Group
#             vaccinated_population = monthly_vaccination_rate * population_per_age_group
            
#             # Assume Half Received First Dose and Half Received Second Dose
#             aij = vaccinated_population / 2  # First dose
#             bij = vaccinated_population / 2  # Second dose
#             cij = population_per_age_group - aij - bij  # Unvaccinated
            
#             # Calculate S''i Based on Vaccine Efficacy
#             S_double_prime_i = (vaccine_efficacy_second_dose[i] * bij +
#                                  vaccine_efficacy_first_dose[i] * aij)
            
#             # Calculate Si (Effective Susceptible)
#             Si = aij + bij + cij - S_double_prime_i  
            
#             # Age-Specific Probability
#             pi = age_group_probabilities_icu[i]
            
#             # Normalize Si by Total Population in Age Group
#             Si_normalized = Si / population_per_age_group
            
#             # Weighted Sum
#             Si_sum += pi * Si_normalized
        
#         # Vax Index for the Day
#         vax_index = Si_sum
#         vax_index_list.append(vax_index)
    
#     # Add Vax Index to the DataFrame
#     df['Vax_index'] = vax_index_list
#     print("Calculated Vax_index for all dates.")
#     return df

# # Calculate Vax Index
# data = calculate_vax_index(data)
# print("Data with Vax_index:")
# display(data.head())

# # --------------------------- #
# #        Data Visualization   #
# # --------------------------- #

# # Create Subplots with 5 Rows
# fig = sp.make_subplots(
#     rows=5, cols=1, 
#     shared_xaxes=True, 
#     subplot_titles=(
#         'New Hospital Admissions', 
#         'Current Hospital Cases',
#         'Mechanical Ventilator Bed Usage',
#         'New COVID-19 Cases',
#         'Vax Index'
#     )
# )

# # Plot New Hospital Admissions
# fig.add_trace(
#     go.Scatter(
#         x=data['date'], 
#         y=data['newAdmissions'], 
#         line=dict(color='brown', width=2), 
#         name='New Admissions'
#     ), 
#     row=1, col=1
# )

# # Plot Current Hospital Cases
# fig.add_trace(
#     go.Scatter(
#         x=data['date'], 
#         y=data['hospitalCases'], 
#         line=dict(color='green', width=2), 
#         name='Hospital Cases'
#     ), 
#     row=2, col=1
# )

# # Plot Mechanical Ventilator Beds Usage
# fig.add_trace(
#     go.Scatter(
#         x=data['date'], 
#         y=data['covidOccupiedMVBeds'], 
#         line=dict(color='blue', width=2), 
#         name='Ventilator Beds'
#     ), 
#     row=3, col=1
# )

# # Plot New COVID-19 Cases
# fig.add_trace(
#     go.Scatter(
#         x=data['date'], 
#         y=data['new_confirmed'], 
#         line=dict(color='orange', width=2), 
#         name='New Cases'
#     ), 
#     row=4, col=1
# )

# # Plot Vax Index
# fig.add_trace(
#     go.Scatter(
#         x=data['date'], 
#         y=data['Vax_index'], 
#         line=dict(color='purple', width=2), 
#         name='Vax Index'
#     ), 
#     row=5, col=1
# )

# # Update Layout
# fig.update_layout(
#     height=1200, 
#     width=900, 
#     title_text="COVID-19 Data Visualization for England",
#     showlegend=True
# )

# fig.show()

# # --------------------------- #
# #       Utility Functions     #
# # --------------------------- #

# def format_plot(
#     fig, legends=None, xlabel="Time", ylabel="Value", title="", font_size=15
# ):
#     """
#     Format a Plotly figure with common styling options.
    
#     Args:
#         fig (plotly.graph_objects.Figure): The Plotly figure to format.
#         legends (list, optional): List of legend names to update.
#         xlabel (str): Label for the x-axis.
#         ylabel (str): Label for the y-axis.
#         title (str): Title of the plot.
#         font_size (int): Font size for labels and legends.
    
#     Returns:
#         plotly.graph_objects.Figure: The formatted Plotly figure.
#     """
#     if legends:
#         names = cycle(legends)
#         fig.for_each_trace(lambda t: t.update(name=next(names)))
#     fig.update_layout(
#         autosize=False,
#         width=900,
#         height=500,
#         title_text=title,
#         title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
#         titlefont={"size": 20},
#         legend_title=None,
#         legend=dict(
#             font=dict(size=font_size),
#             orientation="h",
#             yanchor="bottom",
#             y=0.98,
#             xanchor="right",
#             x=1,
#         ),
#         yaxis=dict(
#             title_text=ylabel,
#             titlefont=dict(size=font_size),
#             tickfont=dict(size=font_size),
#         ),
#         xaxis=dict(
#             title_text=xlabel,
#             titlefont=dict(size=font_size),
#             tickfont=dict(size=font_size),
#         ),
#     )
#     return fig

# def mase(actual, predicted, insample_actual):
#     """
#     Calculate the Mean Absolute Scaled Error (MASE).
    
#     Args:
#         actual (np.array): Actual values.
#         predicted (np.array): Predicted values.
#         insample_actual (np.array): In-sample actual values for scaling.
    
#     Returns:
#         float: The MASE value.
#     """
#     mae_insample = np.mean(np.abs(np.diff(insample_actual)))
#     mae_outsample = np.mean(np.abs(actual - predicted))
#     return mae_outsample / mae_insample

# def forecast_bias(actual, predicted):
#     """
#     Calculate the forecast bias.
    
#     Args:
#         actual (np.array): Actual values.
#         predicted (np.array): Predicted values.
    
#     Returns:
#         float: The forecast bias.
#     """
#     return np.mean(predicted - actual)

# def plot_forecast(pred_df, forecast_columns, forecast_display_names=None, save_path=None):
#     """
#     Plot the forecasted values against actual values.
    
#     Args:
#         pred_df (pd.DataFrame): DataFrame containing actual and predicted values.
#         forecast_columns (list): List of columns with forecasted data.
#         forecast_display_names (list, optional): Display names for the forecasted columns.
#         save_path (str, optional): Path to save the plot.
    
#     Returns:
#         plotly.graph_objects.Figure: The forecast plot.
#     """
#     if forecast_display_names is None:
#         forecast_display_names = forecast_columns
#     else:
#         assert len(forecast_columns) == len(forecast_display_names)

#     mask = ~pred_df[forecast_columns[0]].isnull()
#     colors = px.colors.qualitative.Set2
#     act_color = colors[0]
#     colors = cycle(colors[1:])

#     fig = go.Figure()

#     # Actual Data Plot
#     fig.add_trace(
#         go.Scatter(
#             x=pred_df[mask].index,
#             y=pred_df['covidOccupiedMVBeds'][mask],
#             mode="lines",
#             line=dict(color=act_color, width=2),
#             name="Actual COVID-19 MVBeds Trends",
#         )
#     )

#     # Predicted Data Plots
#     for col, display_col in zip(forecast_columns, forecast_display_names):
#         fig.add_trace(
#             go.Scatter(
#                 x=pred_df[mask].index,
#                 y=pred_df.loc[mask, col],
#                 mode="lines+markers",
#                 marker=dict(size=4),
#                 line=dict(color=next(colors), width=2),
#                 name=display_col,
#             )
#         )

#     return fig

# def highlight_abs_min(s, props=""):
#     """
#     Highlight the absolute minimum in a series.
    
#     Args:
#         s (pd.Series): The series to evaluate.
#         props (str): Properties to apply for highlighting.
    
#     Returns:
#         np.array: Array indicating where to apply the properties.
#     """
#     return np.where(s == np.nanmin(np.abs(s.values)), props, "")

# # --------------------------- #
# #        Data Splitting       #
# # --------------------------- #

# # Find the Minimum and Maximum Dates
# min_date = data['date'].min()
# max_date = data['date'].max()

# print(f"Minimum Date: {min_date}")
# print(f"Maximum Date: {max_date}")

# # Calculate the Date Ranges for Train, Validation, and Test Sets
# date_range = max_date - min_date
# train_end = min_date + pd.Timedelta(days=int(date_range.days * 0.75))
# val_end = train_end + pd.Timedelta(days=int(date_range.days * 0.10))

# # Split the Data into Train, Validation, and Test Sets Based on the Date Ranges
# train = data[data['date'] < train_end]
# val = data[(data['date'] >= train_end) & (data['date'] < val_end)]
# test = data[data['date'] >= val_end]

# # Calculate the Percentage of Dates in Each Dataset
# total_samples = len(data)
# train_percentage = len(train) / total_samples * 100
# val_percentage = len(val) / total_samples * 100
# test_percentage = len(test) / total_samples * 100

# print(f"# of Training samples: {len(train)} | # of Validation samples: {len(val)} | # of Test samples: {len(test)}")
# print(f"Percentage of Dates in Train: {train_percentage:.2f}% | Percentage of Dates in Validation: {val_percentage:.2f}% | Percentage of Dates in Test: {test_percentage:.2f}%")
# print(f"Max Date in Train: {train.date.max()} | Min Date in Validation: {val.date.min()} | Min Date in Test: {test.date.min()}")

# # Drop the 'areaName' Column as It's No Longer Needed
# train = train.drop('areaName', axis=1)
# val = val.drop('areaName', axis=1)
# test = test.drop('areaName', axis=1)

# # Set 'date' as Index
# train.set_index("date", inplace=True)
# val.set_index("date", inplace=True)
# test.set_index("date", inplace=True)

# # Concatenate the DataFrames
# sample_df = pd.concat([train, val, test])

# # Convert Feature Columns to float32
# for col in sample_df.columns:
#     if col != "type":
#         sample_df[col] = sample_df[col].astype("float32")

# print("Sample DataFrame Head:")
# display(sample_df.head())

# # Select Relevant Columns
# columns_to_select = [
#     'covidOccupiedMVBeds',
#     'hospitalCases',
#     'newAdmissions',
#     'new_confirmed',
#     'new_deceased',
#     # 'month',
#     # 'day_of_week',
#     'Vax_index'
# ]
# sample_df = sample_df[columns_to_select]
# print("Selected Columns DataFrame Head:")
# display(sample_df.head())

# # Rearrange Columns to Place 'covidOccupiedMVBeds' Last
# cols = list(sample_df.columns)
# cols.remove("covidOccupiedMVBeds")
# sample_df = sample_df[cols + ["covidOccupiedMVBeds"]]
# print("Rearranged Columns DataFrame Head:")
# display(sample_df.head())

# # Define Target Variable
# target = "covidOccupiedMVBeds"
# pred_df = pd.concat([train[[target]], val[[target]]])

# # Display DataFrame Information
# print("Sample DataFrame Info:")
# display(sample_df.info())

# # --------------------------- #
# #    Data Module & Model      #
# # --------------------------- #

# # Determine Optimal Number of Workers
# num_workers = min(15, multiprocessing.cpu_count())
# print(f"Setting num_workers to: {num_workers}")

# # Initialize Data Module with Increased num_workers
# datamodule = TimeSeriesDataModule(
#     data=sample_df,
#     n_val=len(val),
#     n_test=len(test),
#     window=7, 
#     horizon=1,  
#     normalize="global",  
#     batch_size=32,
#     num_workers=num_workers,
# )
# datamodule.setup()

# # Configure the LSTM Model
# rnn_config = SingleStepRNNConfig(
#     rnn_type="LSTM",  # Changed from "RNN" to "LSTM"
#     input_size=len(columns_to_select),  # Number of input features
#     hidden_size=64,
#     num_layers=3,
#     bidirectional=False,
#     learning_rate=1e-3
# )
# model = SingleStepRNNModel(rnn_config)
# model.float()

# # --------------------------- #
# #        Model Training       #
# # --------------------------- #

# # Initialize Trainer with Adjusted log_every_n_steps and Early Stopping
# trainer = pl.Trainer(
#     min_epochs=10,  # Increased min_epochs for better training
#     max_epochs=100,
#     callbacks=[
#         pl.callbacks.EarlyStopping(monitor="valid_loss", patience=10)  # Increased patience
#     ],
#     log_every_n_steps=10,  # Adjusted logging interval
#     accelerator='gpu' if gpu_available else 'cpu',
#     devices=1 if gpu_available else None,
#     precision=16 if gpu_available else 32,  # Use mixed precision if GPU is available
# )

# # Fit the Model
# trainer.fit(model, datamodule)

# # Remove Artifacts Created During Training
# shutil.rmtree("lightning_logs")

# # --------------------------- #
# #        Evaluation           #
# # --------------------------- #

# # Initialize Metric Record
# metric_record = []

# # Get Predictions for Test Set
# predictions = trainer.predict(model, datamodule.test_dataloader())
# predictions = torch.cat(predictions).squeeze().detach().cpu().numpy()

# # Denormalize Predictions Using Data Module's Statistics
# # Updated to remove [target] indexing since mean and std are scalars
# pred_denormalized = predictions * datamodule.train.std + datamodule.train.mean

# # Get Actual Values from Test Set
# actuals = test[target].values

# # Calculate Metrics
# metrics = {
#     "Algorithm": rnn_config.rnn_type,
#     "MAE": mae(actuals, pred_denormalized),
#     "MSE": mse(actuals, pred_denormalized),
#     "MASE": mase(actuals, pred_denormalized, train[target].values),
#     "Forecast Bias": forecast_bias(actuals, pred_denormalized),
# }

# # Format Metrics for Display
# value_formats = ["{}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.2f}"]
# metrics_formatted = {key: fmt.format(value) for key, value, fmt in zip(metrics.keys(), metrics.values(), value_formats)}
# metric_record.append(metrics_formatted)
# print("Evaluation Metrics:")
# print(metrics_formatted)

# # --------------------------- #
# #    Forecast Visualization   #
# # --------------------------- #

# # Create DataFrame with Predictions
# pred_df = pd.DataFrame({"Vanilla LSTM": pred_denormalized}, index=test.index)
# pred_df = test[[target]].join(pred_df)

# # Plot the Forecast
# fig = plot_forecast(
#     pred_df, 
#     forecast_columns=["Vanilla LSTM"], 
#     forecast_display_names=["Vanilla LSTM"]
# )

# # Define Plot Title with Metrics
# title = (f"{rnn_config.rnn_type}: MAE: {metrics['MAE']:.4f} | "
#          f"MSE: {metrics['MSE']:.4f} | MASE: {metrics['MASE']:.4f} | "
#          f"Bias: {metrics['Forecast Bias']:.2f}")

# # Format and Display the Plot
# fig = format_plot(fig, title=title)
# fig.update_xaxes(type="date", range=[min_date + pd.Timedelta(days=365), max_date])
# fig.show()


# residuals = actuals - pred_denormalized
# plt.figure(figsize=(10, 6))
# sns.histplot(residuals, kde=True)
# plt.title("Residuals Distribution")
# plt.xlabel("Residual")
# plt.ylabel("Frequency")
# plt.show()

# # Plot residuals over time
# plt.figure(figsize=(15, 5))
# plt.plot(test.index, residuals, label='Residuals')
# plt.axhline(0, color='red', linestyle='--')
# plt.title("Residuals Over Time")
# plt.xlabel("Date")
# plt.ylabel("Residual")
# plt.legend()
# plt.show()


# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from tqdm.notebook import tqdm
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio
import plotly.express as px
from itertools import cycle
pio.templates.default = "plotly_white"
import tensorflow as tf
import torch
import random
import pytorch_lightning as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import holidays
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.abspath('../'))

from src.utils import plotting_utils
from src.dl.dataloaders import TimeSeriesDataModule
from src.dl.multivariate_models import SingleStepRNNConfig, SingleStepRNNModel

tqdm.pandas()

torch.set_float32_matmul_precision("high")
# pl.seed_everything(100)

seed = 100
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# %%
# Check if a GPU is available
source_data = Path("../data/")
if tf.config.list_physical_devices("GPU"):
    print("GPU is available")
else:
    print("GPU is not available")

# %%
# Load the data
data = pd.read_csv(source_data / "processed" / "merged_nhs_covid_data.csv")
data.head()

# %%
# list all the unique values in the 'areaName' column
data['areaName'].unique()

# %%
# Aggregate data to a single 'England' region
data = data.groupby('date').agg({
    'covidOccupiedMVBeds': 'sum',
    'cumAdmissions': 'sum',
    'hospitalCases': 'sum',
    'newAdmissions': 'sum',
    'new_confirmed': 'sum',
    'new_deceased': 'sum',
    'cumulative_confirmed': 'sum',
    'cumulative_deceased': 'sum',
    'population': 'sum',
    'openstreetmap_id': 'first',
    'latitude': 'first',
    'longitude': 'first'
}).reset_index()
data['areaName'] = 'England'

# Create timeseries features
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.dayofweek

# %%
# Vax Index Calculation Function
def calculate_vax_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Vax index based on vaccination rates and efficacy across age groups.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'date', 'cumAdmissions', 'cumulativeCases', and other relevant columns.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'Vax_index' column.
    """
    # Constants
    total_population = 60_000_000
    number_of_age_groups = 5
    vaccine_efficacy_first_dose = [0.89, 0.427, 0.76, 0.854, 0.75]  # Added 5th value
    vaccine_efficacy_second_dose = [0.92, 0.86, 0.81, 0.85, 0.80]  # Replaced None with 0.80
    age_group_probabilities_icu = [0.01, 0.02, 0.05, 0.1, 0.15]
    monthly_vaccination_rate_increase = 0.05
    vaccination_start_date = pd.Timestamp('2021-01-18')
    
    # Population per age group
    population_per_age_group = total_population / number_of_age_groups
    
    # Initialize Vax index list
    vax_index_list = []
    
    # Monthly vaccination rate (starting from 0)
    monthly_vaccination_rate = 0.0
    
    for index, row in df.iterrows():
        # Increment monthly vaccination rate on the first day of each month after start date
        if row['date'].day == 1 and row['date'] >= vaccination_start_date:
            monthly_vaccination_rate += monthly_vaccination_rate_increase
            # Ensure vaccination rate does not exceed 1
            monthly_vaccination_rate = min(monthly_vaccination_rate, 1.0)
            print(f"Updated monthly vaccination rate to {monthly_vaccination_rate} on {row['date']}")
        
        Si_sum = 0.0
        
        for i in range(number_of_age_groups):
            # Vaccinated population for this age group
            vaccinated_population = monthly_vaccination_rate * population_per_age_group
            
            # Assume half received first dose and half received second dose
            aij = vaccinated_population / 2  # First dose
            bij = vaccinated_population / 2  # Second dose
            cij = population_per_age_group - aij - bij  # Unvaccinated
            
            # Calculate S''i based on vaccine efficacy
            # Ensuring indices match
            S_double_prime_i = (vaccine_efficacy_second_dose[i] * bij +
                                 vaccine_efficacy_first_dose[i] * aij)
            
            # Calculate Si
            Si = aij + bij + cij - S_double_prime_i  # Effective susceptible
            
            # Age-specific probability
            pi = age_group_probabilities_icu[i]
            
            # Normalize Si by total population in age group
            Si_normalized = Si / population_per_age_group
            
            # Weighted sum
            Si_sum += pi * Si_normalized
        
        # Vax index for the day
        vax_index = Si_sum
        vax_index_list.append(vax_index)
    
    # Add Vax index to the dataframe
    df['Vax_index'] = vax_index_list
    print("Calculated Vax_index for all dates.")
    return df

# Calculate Vax index
data = calculate_vax_index(data)
data.head()

# %%
# Create a subplot with 5 rows
fig = sp.make_subplots(rows=5, cols=1, 
                       shared_xaxes=True, 
                       subplot_titles=(
                                     'New Hospital Admissions', 
                                     'Current Hospital Cases',
                                     'Mechanical Ventilator Bed Usage',
                                     'New COVID-19 Cases',
                                     'Vax Index'))

# Plot for New Hospital Admissions
fig.add_trace(go.Scatter(x=data['date'], 
                        y=data['newAdmissions'], 
                        line=dict(color='brown', width=2), 
                        name='New Admissions'), row=1, col=1)

# Plot for Current Hospital Cases
fig.add_trace(go.Scatter(x=data['date'], 
                        y=data['hospitalCases'], 
                        line=dict(color='green', width=2), 
                        name='Hospital Cases'), row=2, col=1)

# Plot for Mechanical Ventilators
fig.add_trace(go.Scatter(x=data['date'], 
                        y=data['covidOccupiedMVBeds'], 
                        line=dict(color='blue', width=2), 
                        name='Ventilator Beds'), row=3, col=1)

# Plot for New Cases
fig.add_trace(go.Scatter(x=data['date'], 
                        y=data['new_confirmed'], 
                        line=dict(color='orange', width=2), 
                        name='New Cases'), row=4, col=1)

# Plot for Vax Index
fig.add_trace(go.Scatter(x=data['date'], 
                        y=data['Vax_index'], 
                        line=dict(color='purple', width=2), 
                        name='Vax Index'), row=5, col=1)

# Update the layout
fig.update_layout(height=1200, 
                 width=800, 
                 title_text="COVID-19 Data visualisation for England",
                 showlegend=True)

fig.show()

# %%
def format_plot(
    fig, legends=None, xlabel="Time", ylabel="Value", title="", font_size=15
):
    if legends:
        names = cycle(legends)
        fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_layout(
        autosize=False,
        width=900,
        height=500,
        title_text=title,
        title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
        titlefont={"size": 20},
        legend_title=None,
        legend=dict(
            font=dict(size=font_size),
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="right",
            x=1,
        ),
        yaxis=dict(
            title_text=ylabel,
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
        ),
        xaxis=dict(
            title_text=xlabel,
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
        ),
    )
    return fig


def mase(actual, predicted, insample_actual):
    mae_insample = np.mean(np.abs(np.diff(insample_actual)))
    mae_outsample = np.mean(np.abs(actual - predicted))
    return mae_outsample / mae_insample


def forecast_bias(actual, predicted):
    return np.mean(predicted - actual)


def plot_forecast(pred_df, forecast_columns, forecast_display_names=None, save_path=None):
    """
    Plot the forecasted values against actual values.
    
    Args:
        pred_df (pd.DataFrame): DataFrame containing actual and predicted values.
        forecast_columns (list): List of columns with forecasted data.
        forecast_display_names (list, optional): Display names for the forecasted columns.
        save_path (str, optional): Path to save the plot.
    
    Returns:
        plotly.graph_objects.Figure: The forecast plot.
    """
    if forecast_display_names is None:
        forecast_display_names = forecast_columns
    else:
        assert len(forecast_columns) == len(forecast_display_names)

    mask = ~pred_df[forecast_columns[0]].isnull()
    colors = px.colors.qualitative.Set2
    act_color = colors[0]
    colors = cycle(colors[1:])

    fig = go.Figure()

    # Actual Data Plot
    fig.add_trace(
        go.Scatter(
            x=pred_df[mask].index,
            y=pred_df['covidOccupiedMVBeds'][mask],  # Fixed column reference
            mode="lines",
            line=dict(color=act_color, width=2),
            name="Actual COVID-19 MVBeds Trends",
        )
    )

    # Predicted Data Plots
    for col, display_col in zip(forecast_columns, forecast_display_names):
        fig.add_trace(
            go.Scatter(
                x=pred_df[mask].index,
                y=pred_df.loc[mask, col],
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(color=next(colors), width=2),
                name=display_col,
            )
        )

    return fig


def highlight_abs_min(s, props=""):
    return np.where(s == np.nanmin(np.abs(s.values)), props, "")

# %%
# Calculate rolling statistics
window_size_7 = 7
window_size_14 = 14

# List of columns for which rolling statistics need to be computed
columns_to_compute = ['covidOccupiedMVBeds', 'hospitalCases', 'newAdmissions', 'Vax_index', 'new_confirmed']

# Compute rolling statistics for each column
for column in columns_to_compute:
    data[f'{column}_rolling_mean_7'] = data[column].rolling(window=window_size_7).mean()
    data[f'{column}_rolling_std_7'] = data[column].rolling(window=window_size_7).std()
    data[f'{column}_rolling_mean_14'] = data[column].rolling(window=window_size_14).mean()
    data[f'{column}_rolling_std_14'] = data[column].rolling(window=window_size_14).std()
    
data.head()

# %%
# Define the enhanced feature engineering function
def engineer_features(df):
    """
    Engineer comprehensive features for ventilator demand forecasting.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing COVID-19 data
        
    Returns:
        pandas.DataFrame: DataFrame with additional engineered features
    """
    # Create a copy to avoid modifying the original DataFrame
    data = df.copy()
    
    # Ensure date is in datetime format
    if isinstance(data['date'].iloc[0], str):
        data['date'] = pd.to_datetime(data['date'])
    
    # Sort by date to ensure proper calculation of time-based features
    data = data.sort_values('date').reset_index(drop=True)
    
    #------------------
    # LOCKDOWN FEATURES
    #------------------
    # Define lockdown periods
    lockdown_dates = {
        'Lockdown 1': {'start': '2020-03-23', 'end': '2020-07-04'},
        'Lockdown 2': {'start': '2020-11-05', 'end': '2020-12-02'},
        'Lockdown 3': {'start': '2021-01-06', 'end': '2021-04-12'}
    }
    
    # Create lockdown indicators
    data['in_lockdown'] = 0
    data['days_since_lockdown_start'] = np.nan
    data['days_until_lockdown_end'] = np.nan
    
    # Create features for each lockdown period
    for lockdown_name, period in lockdown_dates.items():
        start_date = pd.to_datetime(period['start'])
        end_date = pd.to_datetime(period['end'])
        
        # Create binary indicator for this specific lockdown
        lockdown_col = f'in_{lockdown_name.lower().replace(" ", "_")}'
        data[lockdown_col] = ((data['date'] >= start_date) & (data['date'] <= end_date)).astype(int)
        
        # Update the general lockdown indicator
        data.loc[(data['date'] >= start_date) & (data['date'] <= end_date), 'in_lockdown'] = 1
        
        # Calculate days since lockdown start (for dates within and after the lockdown)
        mask_since = data['date'] >= start_date
        data.loc[mask_since, f'days_since_{lockdown_name.lower().replace(" ", "_")}_start'] = (
            (data.loc[mask_since, 'date'] - start_date).dt.days
        )
        
        # Calculate days until lockdown end (for dates within the lockdown)
        mask_until = (data['date'] >= start_date) & (data['date'] <= end_date)
        data.loc[mask_until, f'days_until_{lockdown_name.lower().replace(" ", "_")}_end'] = (
            (end_date - data.loc[mask_until, 'date']).dt.days
        )
        
        # Fill NaN values with -1 for dates before the lockdown period
        data[f'days_since_{lockdown_name.lower().replace(" ", "_")}_start'] = data[f'days_since_{lockdown_name.lower().replace(" ", "_")}_start'].fillna(-1)
        data[f'days_until_{lockdown_name.lower().replace(" ", "_")}_end'] = data[f'days_until_{lockdown_name.lower().replace(" ", "_")}_end'].fillna(-1)
    
    # Calculate days since/until any lockdown
    for i, row in data.iterrows():
        current_date = row['date']
        
        # Find the closest lockdown start date that's in the past
        past_starts = [(pd.to_datetime(period['start']), name) 
                       for name, period in lockdown_dates.items() 
                       if pd.to_datetime(period['start']) <= current_date]
        
        if past_starts:
            closest_past_start = max(past_starts, key=lambda x: x[0])
            data.at[i, 'days_since_lockdown_start'] = (current_date - closest_past_start[0]).days
        else:
            data.at[i, 'days_since_lockdown_start'] = -1
            
        # Find the closest lockdown end date that's in the future
        future_ends = [(pd.to_datetime(period['end']), name) 
                       for name, period in lockdown_dates.items() 
                       if pd.to_datetime(period['start']) <= current_date <= pd.to_datetime(period['end'])]
        
        if future_ends:
            closest_future_end = min(future_ends, key=lambda x: x[0])
            data.at[i, 'days_until_lockdown_end'] = (closest_future_end[0] - current_date).days
        else:
            data.at[i, 'days_until_lockdown_end'] = -1
    
    # Create a feature for time since last lockdown ended
    data['days_since_last_lockdown'] = -1
    for i, row in data.iterrows():
        current_date = row['date']
        
        # Find past lockdown end dates
        past_ends = [(pd.to_datetime(period['end']), name) 
                    for name, period in lockdown_dates.items() 
                    if pd.to_datetime(period['end']) < current_date]
        
        if past_ends and data.at[i, 'in_lockdown'] == 0:
            most_recent_end = max(past_ends, key=lambda x: x[0])
            data.at[i, 'days_since_last_lockdown'] = (current_date - most_recent_end[0]).days
    
    #-------------------------
    # RATE OF CHANGE FEATURES
    #-------------------------
    # 1. Rate of change (daily) for hospital cases
    data['hospitalCases_daily_change'] = data['hospitalCases'].diff()
    data['hospitalCases_pct_change'] = data['hospitalCases'].pct_change() * 100
    
    # 2. Rate of change (daily) for new admissions
    data['newAdmissions_daily_change'] = data['newAdmissions'].diff()
    data['newAdmissions_pct_change'] = data['newAdmissions'].pct_change() * 100
    
    # 3. Rate of change in ventilator usage
    data['vent_daily_change'] = data['covidOccupiedMVBeds'].diff()
    data['vent_pct_change'] = data['covidOccupiedMVBeds'].pct_change() * 100
    
    # 4. Rate of change in confirmed cases
    if 'new_confirmed' in data.columns:
        data['confirmed_daily_change'] = data['new_confirmed'].diff()
        data['confirmed_pct_change'] = data['new_confirmed'].pct_change() * 100
    
    #-------------------
    # MOMENTUM FEATURES
    #-------------------
    # Multi-day momentum features (change over 3 and 7 days)
    for col in ['hospitalCases', 'newAdmissions', 'covidOccupiedMVBeds']:
        data[f'{col}_3day_momentum'] = data[col].diff(3)
        data[f'{col}_7day_momentum'] = data[col].diff(7)
    
    if 'new_confirmed' in data.columns:
        data['new_confirmed_3day_momentum'] = data['new_confirmed'].diff(3)
        data['new_confirmed_7day_momentum'] = data['new_confirmed'].diff(7)
    
    #---------------
    # RATIO FEATURES
    #---------------
    # 1. Percentage of hospital cases requiring ventilation
    data['pct_cases_ventilated'] = (data['covidOccupiedMVBeds'] / data['hospitalCases']) * 100
    
    # 2. Admission to hospital ratio
    data['admission_to_hospital_ratio'] = data['newAdmissions'] / data['hospitalCases']
    
    # 3. Ventilator to hospital ratio
    data['vent_to_hospital_ratio'] = data['covidOccupiedMVBeds'] / data['hospitalCases']
    
    # 4. Admission to ventilator ratio (proxy for severity)
    data['admission_to_vent_ratio'] = data['newAdmissions'] / data['covidOccupiedMVBeds']
    
    #----------------
    # CALENDAR FEATURES
    #----------------
    # 1. Weekend flag (0 for weekday, 1 for weekend)
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
    
    # 2. UK holiday flags
    uk_holidays = holidays.UK()
    data['is_holiday'] = data['date'].apply(lambda x: 1 if x in uk_holidays else 0)
    
    # 3. Day of month
    data['day_of_month'] = data['date'].dt.day
    
    # 4. Week of year
    data['week_of_year'] = data['date'].dt.isocalendar().week.astype(int)
    
    # 5. Quarter
    data['quarter'] = data['date'].dt.quarter
    
    #---------------------
    # PEAK-RELATED FEATURES
    #---------------------
    # Days since most recent peak in key metrics
    def days_since_peak(series):
        result = np.zeros(len(series))
        current_max = series.iloc[0]
        current_max_idx = 0
        
        for i in range(1, len(series)):
            if series.iloc[i] > current_max:
                current_max = series.iloc[i]
                current_max_idx = i
            
            result[i] = i - current_max_idx
        
        return result
    
    data['days_since_vent_peak'] = days_since_peak(data['covidOccupiedMVBeds'])
    data['days_since_hospital_peak'] = days_since_peak(data['hospitalCases'])
    data['days_since_admissions_peak'] = days_since_peak(data['newAdmissions'])
    
    # Days until next peak (forward-looking)
    def days_until_next_peak(series):
        n = len(series)
        result = np.zeros(n)
        
        # First pass to find all peaks
        peaks = []
        for i in range(1, n-1):
            if series.iloc[i] > series.iloc[i-1] and series.iloc[i] > series.iloc[i+1]:
                if len(peaks) == 0 or series.iloc[i] > series.iloc[peaks[-1]]:
                    peaks.append(i)
        
        # Second pass to assign days until next peak
        current_peak_idx = 0
        for i in range(n):
            if current_peak_idx < len(peaks):
                if i < peaks[current_peak_idx]:
                    result[i] = peaks[current_peak_idx] - i
                else:
                    current_peak_idx += 1
                    if current_peak_idx < len(peaks):
                        result[i] = peaks[current_peak_idx] - i
                    else:
                        result[i] = -1  # No more peaks ahead
            else:
                result[i] = -1  # No more peaks ahead
        
        return result
    
    # Apply to smoothed data to avoid minor fluctuations
    if 'covidOccupiedMVBeds_rolling_mean_7' in data.columns:
        data['days_until_next_vent_peak'] = days_until_next_peak(data['covidOccupiedMVBeds_rolling_mean_7'])
    
    #----------------------
    # ACCELERATION FEATURES
    #----------------------
    # Change in the rate of change (acceleration)
    data['hospitalCases_acceleration'] = data['hospitalCases_daily_change'].diff()
    data['vent_acceleration'] = data['vent_daily_change'].diff()
    data['admissions_acceleration'] = data['newAdmissions_daily_change'].diff()
    
    #-------------------
    # TREND RATIO FEATURES
    #-------------------
    # Moving average ratios (Short-term vs long-term trends)
    if all(col in data.columns for col in ['hospitalCases_rolling_mean_7', 'hospitalCases_rolling_mean_14']):
        data['hospital_trend_ratio'] = data['hospitalCases_rolling_mean_7'] / data['hospitalCases_rolling_mean_14']
    
    if all(col in data.columns for col in ['covidOccupiedMVBeds_rolling_mean_7', 'covidOccupiedMVBeds_rolling_mean_14']):
        data['vent_trend_ratio'] = data['covidOccupiedMVBeds_rolling_mean_7'] / data['covidOccupiedMVBeds_rolling_mean_14']
    
    if all(col in data.columns for col in ['newAdmissions_rolling_mean_7', 'newAdmissions_rolling_mean_14']):
        data['admissions_trend_ratio'] = data['newAdmissions_rolling_mean_7'] / data['newAdmissions_rolling_mean_14']
    
    #-------------
    # LAG FEATURES
    #-------------
    # Lag features for key metrics (t-1, t-7, t-14)
    for col in ['covidOccupiedMVBeds', 'hospitalCases', 'newAdmissions']:
        data[f'{col}_lag_1'] = data[col].shift(1)
        data[f'{col}_lag_7'] = data[col].shift(7)
        data[f'{col}_lag_14'] = data[col].shift(14)
    
    if 'new_confirmed' in data.columns:
        data['new_confirmed_lag_1'] = data['new_confirmed'].shift(1)
        data['new_confirmed_lag_7'] = data['new_confirmed'].shift(7)
    
    #-----------------
    # SEASONAL FEATURES
    #-----------------
    # Season indicators (Winter, Spring, Summer, Fall)
    data['season'] = data['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 
                  2 if x in [3, 4, 5] else 
                  3 if x in [6, 7, 8] else 4
    )
    # Convert to one-hot encoding
    season_dummies = pd.get_dummies(data['season'], prefix='season')
    data = pd.concat([data, season_dummies], axis=1)
    
    #---------------
    # WAVE INDICATORS
    #---------------
    # COVID wave indicator using threshold approach
    def identify_waves(series, threshold_multiplier=1.5):
        # Use rolling mean to smooth the data
        smooth = series.rolling(window=14, min_periods=1).mean()
        # Calculate the overall mean
        overall_mean = smooth.mean()
        # Mark as wave when above threshold
        return (smooth > (overall_mean * threshold_multiplier)).astype(int)
    
    data['covid_wave'] = identify_waves(data['covidOccupiedMVBeds'])
    
    # Alternative wave indicator using rate of change
    data['wave_momentum'] = (data['covidOccupiedMVBeds_rolling_mean_7'].diff(7) > 0).astype(int)
    
    #-------------------
    # LOCKDOWN INTERACTION
    #-------------------
    # Interaction features between lockdown and other metrics
    data['lockdown_hospital_interaction'] = data['in_lockdown'] * data['hospitalCases']
    data['lockdown_admission_interaction'] = data['in_lockdown'] * data['newAdmissions']
    data['lockdown_vent_interaction'] = data['in_lockdown'] * data['covidOccupiedMVBeds']
    
    # Lockdown effectiveness metrics (change in key metrics during lockdown periods)
    for col in ['hospitalCases', 'newAdmissions', 'covidOccupiedMVBeds']:
        data[f'{col}_lockdown_effect'] = data[col].diff(7) * data['in_lockdown']
    
    #-----------------
    # COMPOSITE FEATURES
    #-----------------
    # Combined features that might capture complex relationships
    data['composite_hospital_admission_ratio'] = (data['hospitalCases'] + data['newAdmissions']) / data['hospitalCases']
    data['delta_admission_to_ventilator'] = data['vent_daily_change'] - data['newAdmissions_daily_change']
    
    # Dynamic pressure indicator (how fast ventilator usage is growing relative to capacity)
    data['ventilator_pressure'] = data['vent_pct_change'] * data['vent_to_hospital_ratio']
    
    # Systems stress indicator (combining multiple factors)
    data['system_stress'] = (
        data['pct_cases_ventilated'] * 
        data['hospitalCases_pct_change'].clip(lower=0) * 
        (1 / (data['days_since_admissions_peak'] + 1))
    )
    
    # Fill NaN values created by diff and shift operations
    # For ratio columns, forward fill might be better
    ratio_cols = [col for col in data.columns if 'ratio' in col.lower()]
    data[ratio_cols] = data[ratio_cols].fillna(method='ffill')
    
    # For other columns with NaNs, use 0 (assuming start of the series)
    data = data.fillna(0)
    
    return data

# Apply enhanced feature engineering
enhanced_data = engineer_features(data)
print(f"Original data columns: {len(data.columns)}")
print(f"Enhanced data columns: {len(enhanced_data.columns)}")
print(f"New features added: {len(enhanced_data.columns) - len(data.columns)}")

# %%
# Find the minimum and maximum dates
min_date = enhanced_data['date'].min()
max_date = enhanced_data['date'].max()

print("Minimum Date:", min_date)
print("Maximum Date:", max_date)

# Calculate the date ranges for train, val, and test sets
date_range = max_date - min_date
train_end = min_date + pd.Timedelta(days=date_range.days * 0.75)
val_end = train_end + pd.Timedelta(days=date_range.days * 0.10)

# Split the data into train, validation, and test sets based on the date ranges
train = enhanced_data[enhanced_data['date'] < train_end]
val = enhanced_data[(enhanced_data['date'] >= train_end) & (enhanced_data['date'] < val_end)]
test = enhanced_data[enhanced_data['date'] >= val_end]

# Calculate the percentage of dates in each dataset
total_samples = len(enhanced_data)
train_percentage = len(train) / total_samples * 100
val_percentage = len(val) / total_samples * 100
test_percentage = len(test) / total_samples * 100

print(f"# of Training samples: {len(train)} | # of Validation samples: {len(val)} | # of Test samples: {len(test)}")
print(f"Percentage of Dates in Train: {train_percentage:.2f}% | Percentage of Dates in Validation: {val_percentage:.2f}% | Percentage of Dates in Test: {test_percentage:.2f}%")
print(f"Max Date in Train: {train.date.max()} | Min Date in Validation: {val.date.min()} | Min Date in Test: {test.date.min()}")

# %%
# drop the 'areaName' column as it's no longer needed
train = train.drop('areaName', axis=1)
val = val.drop('areaName', axis=1)
test = test.drop('areaName', axis=1)

# %%
# Convert date to datetime and set as index
train.set_index("date", inplace=True)
test.set_index("date", inplace=True)
val.set_index("date", inplace=True)

# %%
# Concatenate the DataFrames
sample_df = pd.concat([train, val, test])

# Convert feature columns to float32
# Exclude the 'type' column from conversion as it's a string column
for col in sample_df.columns:
    if col != "type":
        sample_df[col] = sample_df[col].astype("float32")

sample_df.head()

# %%
# Select the most relevant features for the model
base_features = [
    'covidOccupiedMVBeds_rolling_mean_7',
    'hospitalCases',
    'newAdmissions',
    'new_confirmed', 
    'Vax_index'
]

# Add lockdown features
lockdown_features = [
    'in_lockdown',
    'days_since_lockdown_start',
    'days_until_lockdown_end',
    'days_since_last_lockdown',
    'lockdown_hospital_interaction',
    'lockdown_admission_interaction',
    'lockdown_vent_interaction'
]

# Add ratio and momentum features
advanced_features = [
    'vent_to_hospital_ratio',
    'pct_cases_ventilated',
    'hospitalCases_3day_momentum',
    'vent_3day_momentum',
    'vent_lag_1',
    'vent_lag_7',
    'system_stress',
    'admission_to_vent_ratio'
]

# Add calendar features
calendar_features = [
    'year',
    'month',
    'day_of_week',
    'is_weekend',
    'is_holiday'
]

# Combine all features
selected_features = base_features + lockdown_features + advanced_features + calendar_features

# Filter to only include features that exist in the dataframe
selected_features = [f for f in selected_features if f in sample_df.columns]

# Select the features and the target variable
selected_df = sample_df[selected_features + ['covidOccupiedMVBeds']]

# %%
# Ensure the target column is the last column
cols = list(selected_df.columns)
if 'covidOccupiedMVBeds' in cols:
    cols.remove('covidOccupiedMVBeds')
    selected_df = selected_df[cols + ['covidOccupiedMVBeds']]

# %%
target = "covidOccupiedMVBeds"
pred_df = pd.concat([train[[target]], val[[target]]])

# %%
# Print info about the selected dataframe
selected_df.info()

# %%
# Create a TimeSeriesDataModule for the model
# Create the datamodule with the selected features
datamodule = TimeSeriesDataModule(
    data=selected_df,
    n_val=val.shape[0],
    n_test=test.shape[0],
    window=7, 
    horizon=1,  
    normalize="global",  
    batch_size=32,
    num_workers=0,
)
datamodule.setup()

# Get the actual input size from the datamodule
actual_input_size = datamodule.train_dataloader().dataset[0][0].shape[1]
print(f"Actual input size from dataloader: {actual_input_size}")

# Configure the RNN model with the correct input size
rnn_config = SingleStepRNNConfig(
    rnn_type="RNN",
    input_size=actual_input_size,  # Use the actual input size
    hidden_size=64,
    num_layers=3,
    bidirectional=False,
    learning_rate=1e-3
)
model = SingleStepRNNModel(rnn_config)
model.float()

# %%
# Train the model
trainer = pl.Trainer(
    min_epochs=5,
    max_epochs=100,
    callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=5)],
)
trainer.fit(model, datamodule)

# %%
# Removing artifacts created during training
import shutil
if os.path.exists("lightning_logs"):
    shutil.rmtree("lightning_logs")

# %%
# Evaluate the model and calculate metrics
metric_record = []

# Get predictions for test set
pred = trainer.predict(model, datamodule.test_dataloader())
pred = torch.cat(pred).squeeze().detach().numpy()

# Denormalize predictions using datamodule's statistics
pred = pred * datamodule.train.std + datamodule.train.mean

# Get actual values from test set
actuals = test[target].values

# Calculate metrics
metrics = {
    "Algorithm": rnn_config.rnn_type,
    "MAE": mean_absolute_error(actuals, pred),
    "MSE": mean_squared_error(actuals, pred),
    "RMSE": np.sqrt(mean_squared_error(actuals, pred)),
    "MAPE": mean_absolute_percentage_error(actuals, pred),
    "MASE": mase(actuals, pred, train[target].values),
    "Forecast Bias": forecast_bias(actuals, pred),
}

# Format metrics for display
value_formats = ["{}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.2f}"]
metrics = {key: format_.format(value) for key, value, format_ in zip(metrics.keys(), metrics.values(), value_formats)}
metric_record.append(metrics)
print(metrics)

# %%
# Create DataFrame with predictions
pred_df = pd.DataFrame({"Enhanced RNN": pred}, index=test.index)
pred_df = test[['covidOccupiedMVBeds']].join(pred_df)

# Prepare for plotting - calculate trend differences if needed
if not hasattr(pred_df, 'covidOccupiedMVBeds_trend_diff'):
    pred_df['covidOccupiedMVBeds_trend_diff'] = pred_df['covidOccupiedMVBeds']

# Plot the forecast
fig = plot_forecast(pred_df, forecast_columns=["Enhanced RNN"], forecast_display_names=["Enhanced RNN with Lockdown Features"])
title = f"{rnn_config.rnn_type}: MAE: {metrics['MAE']} | MSE: {metrics['MSE']} | MASE: {metrics['MASE']} | Bias: {metrics['Forecast Bias']}"
fig = format_plot(fig, title=title)
fig.update_xaxes(type="date")
fig.show()

# %%
# Plot feature importance by perturbing the input data
feature_importance = {}

for feature in selected_features:
    # Create a copy of the test data
    test_perturbed = test.copy()
    
    # Shuffle the feature values
    test_perturbed[feature] = np.random.permutation(test_perturbed[feature].values)
    
    # Prepare data for prediction
    perturbed_data = selected_df.copy()
    perturbed_data.loc[test_perturbed.index, feature] = test_perturbed[feature]
    
    # Create a new datamodule with the perturbed data
    perturbed_datamodule = TimeSeriesDataModule(
        data=perturbed_data,
        n_val=val.shape[0],
        n_test=test.shape[0],
        window=7,
        horizon=1,
        normalize="global",
        batch_size=32,
        num_workers=0,
    )
    perturbed_datamodule.setup()
    
    # Make predictions with the perturbed data
    perturbed_pred = trainer.predict(model, perturbed_datamodule.test_dataloader())
    perturbed_pred = torch.cat(perturbed_pred).squeeze().detach().numpy()
    perturbed_pred = perturbed_pred * perturbed_datamodule.train.std + perturbed_datamodule.train.mean
    
    # Calculate the increase in error due to perturbation
    original_mse = mean_squared_error(actuals, pred)
    perturbed_mse = mean_squared_error(actuals, perturbed_pred)
    
    # Feature importance is the percentage increase in error
    importance = (perturbed_mse - original_mse) / original_mse * 100
    feature_importance[feature] = importance

# Sort features by importance
feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(list(feature_importance.keys()), list(feature_importance.values()))
plt.xlabel('Percentage Increase in MSE')
plt.title('Feature Importance: Impact on Model Performance')
plt.tight_layout()
plt.show()
