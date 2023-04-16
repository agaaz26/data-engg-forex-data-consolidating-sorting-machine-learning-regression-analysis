# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 12:30:45 2023

@author: agaaz
"""

import pandas as pd
import pymongo
import os
import mysql.connector
from pycaret.regression import *

# Connect to MongoDB
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['my_database9']

# Connect to MySQL
cnx = mysql.connector.connect(
    host="localhost",
    user="agaaz26",
    password="mypassword",
    database="mysql_database9"
)
cursor = cnx.cursor()

# Read each CSV file into a pandas df
folder_path = "D:\HW3 Files"

# Create an empty dictionary to hold the dataframes
dfs = {}

# Loop through each CSV file in the folder
for file in os.listdir(folder_path):
    # Read the CSV file into a dataframe
    df = pd.read_csv(os.path.join(folder_path, file))
    # Store the dataframe in the dictionary, with the file name as the key
    dfs[file] = df
 
# Got these dates from querying mongoDB after creating it and then went back to add these before the creation function
starting_datetime = pd.to_datetime("2010-01-03 17:00:00")
ending_datetime = pd.to_datetime("2023-01-24 23:00:00")

# Loop through each dataframe in the dictionary
for key, df in dfs.items():
    # Convert the datetime column to a pandas datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])
    # Group the data by hour and calculate the VWAP and liquidity for each hour
    hourly_df = df.groupby(pd.Grouper(key='datetime', freq='H')).agg({'v': 'sum', 'vw': 'mean'})
    hourly_df = hourly_df.reset_index()
    hourly_df = hourly_df.rename(columns={'datetime': 'hour', 'v': 'liquidity', 'vw': 'vwap' })
    hourly_df['hour'] = pd.date_range(start=starting_datetime, periods=len(hourly_df), freq='H')
    hourly_df = hourly_df.loc[hourly_df['hour'] <= ending_datetime]
    # Insert the data into MongoDB
    db[key[:-4]].insert_many(hourly_df.to_dict('records'))

     # Create a table in the MySQL database for this dataframe, if it doesn't exist
    table_name = key[:-4]
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS `{table_name}` (
        `id` INT AUTO_INCREMENT PRIMARY KEY,
        `hour` DATETIME,
        `liquidity` FLOAT,
        `vwap` FLOAT
    );
    """
    cursor.execute(create_table_query)
    cnx.commit()

    # Insert the data into MySQL
    insert_query = f"INSERT INTO `{table_name}` (`hour`, `liquidity`, `vwap`) VALUES (%s, %s, %s)"
    for _, row in hourly_df.iterrows():
        cursor.execute(insert_query, (row['hour'], row['liquidity'], row['vwap']))
        cnx.commit()

# Close the MySQL connection
cursor.close()
cnx.close()

# Get all collections in the database
collections = db.list_collection_names()

#part B

# Define a function to calculate the volatility
def calc_volatility(data):
    return (data.max() - data.min()) / data.mean()

def calc_keltner_channels(data):
    # Calculate the mean value and volatility from the previous period of 6 hours
    prev_mean = data['vwap_mean'].shift(1)
    prev_volatility = data['vwap_calc_volatility'].shift(1)

    # Calculate the upper and lower bands for each value of n from 1 to 100
    for n in range(1, 101):
        data[f'upper_{n}'] = prev_mean + n * 0.025 * prev_volatility
        data[f'lower_{n}'] = prev_mean - n * 0.025 * prev_volatility

    # Calculate the number of times vwap_mean crosses each keltner band in the given period
    num_crosses = pd.DataFrame()
    for n in range(1, 101):
        upper_band_crosses = (data['vwap_mean'] > data[f'upper_{n}']).astype(int)
        lower_band_crosses = (data['vwap_mean'] < data[f'lower_{n}']).astype(int)
        num_crosses[f'num_crosses_{n}'] = upper_band_crosses.diff() + lower_band_crosses.diff()

    # Calculate the FD values for each period
    data['FD'] = num_crosses.sum(axis=1) / (data['vwap_max'] - data['vwap_min'])

# Drop the intermediate upper and lower band columns
    for n in range(1, 101):
        data.drop(columns=[f'upper_{n}', f'lower_{n}'], inplace=True)

    return data['FD']

# create an empty dictionary to hold separate dataframes for each collection
resampled_dfs = {}


# Loop through each collection
for collection_name in collections:
    # Read the data from the collection into a pandas dataframe
    cursor = db[collection_name].find()
    df = pd.DataFrame(list(cursor))
    df['hour'] = pd.to_datetime(df['hour'])
    
    # Define the frequency of the resampled data
    freq = '6H'
    
    # Resample the data to the defined frequency, calculating the VWAP, liquidity, volatility, max and min values for each period
    resampled_df = df.set_index('hour').resample(freq).agg({'vwap': ['mean', 'max', 'min' , calc_volatility], 'liquidity': 'sum' })
    
    # Sort the resampled dataframe in ascending order by datetime
    resampled_df.sort_index(inplace=True)
    
    #store resampled data in the dictionary with the currency pair name
    resampled_dfs[collection_name] = resampled_df
    
    # Flatten the multi-level column index
    resampled_df.columns = ['_'.join(col).strip() for col in resampled_df.columns.values]
    
    # Create a new column for the timestamp
    resampled_df['timestamp'] = resampled_df.index
    
    # Calculate FD and store it as a column in the resampled data
    resampled_df['FD'] = calc_keltner_channels(resampled_df)

    # Save the resampled data to a CSV file
    resampled_dfs[collection_name].to_csv(f"{collection_name}_keltner_channels.csv", index=False)
    
import numpy as np

# Function to drop infinite values from a dataframe
def drop_inf_values(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# Apply the function to all dataframes in the resampled_dfs dictionary
for key, value in resampled_dfs.items():
    resampled_dfs[key] = drop_inf_values(value)
    
#%%
# Loop through each resampled dataframe in the dictionary
for collection_name, resampled_df in resampled_dfs.items():
    # Drop any rows with missing and 0 values
    resampled_df = resampled_df.dropna()
    resampled_df = resampled_df[resampled_df["FD"] != 0]

    # Set up the PyCaret regression experiment
    exp_reg = setup(
        data=resampled_df,
        target='vwap_mean',
        train_size=0.7,  # 70% training data, 30% testing data
        fold_strategy='timeseries',  # Time series cross-validation
        numeric_features=['vwap_max', 'vwap_min', 'vwap_calc_volatility', 'liquidity_sum', 'FD'],
        profile=False,  # Suppress the confirmation prompt
        verbose=False,  # Suppress output during setup
        session_id=42  # Set random seed for reproducibility
    )

    # Train and compare all regression models
    best_model = compare_models(include=None)  # Add include=None to consider all available models

    results = pull()
    
    results.to_csv(f"{collection_name}_pycaret.csv")




