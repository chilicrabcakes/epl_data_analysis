# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:36:53 2018

@author: hadoop
"""

import pandas as pd
import numpy as np

# EPL Data analysis
# Taking data from www.football-data.co.uk

folder_path = 'your/folder/path/here/'

# Getting data
df_2018 = pd.read_csv(folder_path + '2018-2019.csv')
df_2017 = pd.read_csv(folder_path + '2017-2018.csv')
df_2016 = pd.read_csv(folder_path + '2016-2017.csv')
df_2015 = pd.read_csv(folder_path + '2015-2016.csv')

# Concatenating the data
df = pd.concat([df_2018, df_2017, df_2016, df_2015] , axis=0, join='inner', ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')

# Columns that need to be removed (changed into aggregate averages)
# FTHG, FTAG, FTR (result - label), HTHG, HTAG, HTR, HS, AS, HST, AST 
# HF, AF, HC, AC, HY, AY, HR, AR

df['FTHG_AGG'] = np.nan;
df['FTAG_AGG'] = np.nan;
df['HTHG_AGG'] = np.nan;
df['HTAG_AGG'] = np.nan;
df['HS_AGG'] = np.nan;
df['AS_AGG'] = np.nan;
df['HST_AGG'] = np.nan;
df['AST_AGG'] = np.nan;
df['HF_AGG'] = np.nan;
df['AF_AGG'] = np.nan;
df['HC_AGG'] = np.nan;
df['AC_AGG'] = np.nan;
df['HY_AGG'] = np.nan;
df['AY_AGG'] = np.nan;
df['HR_AGG'] = np.nan;
df['AR_AGG'] = np.nan;

# Aggregates the column values using time (need better commenting for this function)
# aggregator: A function that does the required data processing for a given
# type of information. In the EPL data, we cannot use data points such as FTHG, FTAG
# and so on as these are values observed DURING the game. For match predictions to 
# be made before the game, using these values would make all our predictions rubbish.
# So, the way to deal with this is to find the average values of match data before
# the match - which is what this function does. 
def aggregator (home, home_column, away_column, output_column):
    for index, row in df.iterrows():
        hometeam = row[home]
        df_slice = df[df['Date'] < row['Date']]
        if (len(df_slice) == 0): 
            continue
        sum_series = df_slice.loc[(df_slice['HomeTeam'] == hometeam)][home_column].append(df_slice.loc[(df['AwayTeam'] == hometeam)][away_column])
        df[output_column].iat[index] = sum_series.mean()

aggregator('HomeTeam','FTHG','FTAG','FTHG_AGG')
aggregator('AwayTeam','FTHG','FTAG','FTAG_AGG')
aggregator('HomeTeam','HTHG','HTAG','HTHG_AGG')
aggregator('AwayTeam','HTHG','HTAG','HTAG_AGG')
aggregator('HomeTeam','HS','AS','HS_AGG')
aggregator('AwayTeam','HS','AS','AS_AGG')
aggregator('HomeTeam','HST','AST','HST_AGG')
aggregator('AwayTeam','HST','AST','AST_AGG')
aggregator('HomeTeam','HF','AF','HF_AGG')
aggregator('AwayTeam','HF','AF','AF_AGG')
aggregator('HomeTeam','HC','AC','HC_AGG')
aggregator('AwayTeam','HC','AC','AC_AGG')
aggregator('HomeTeam','HY','AY','HY_AGG')
aggregator('AwayTeam','HY','AY','AY_AGG')
aggregator('HomeTeam','HR','AR','HR_AGG')
aggregator('AwayTeam','HR','AR','AR_AGG')

# Data cleaning: Clearing out nan values, dropping post-game data
df = df.sort_values(by=['Date'])
final_df = df.drop(columns=['Div','Date','Referee','HTR','FTHG','FTAG','HTHG','HTAG','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR'])
final_df = final_df.dropna(axis=0, how='any')

# Label Encoding: Converting string data into integer values for 
# classification
from sklearn import preprocessing
le_team = preprocessing.LabelEncoder()
le_res = preprocessing.LabelEncoder()
le_team.fit(final_df['HomeTeam'])
final_df['HomeTeam'] = le_team.transform(final_df['HomeTeam'])
final_df['AwayTeam'] = le_team.transform(final_df['AwayTeam'])
final_df['FTR'] = le_res.fit_transform(final_df['FTR'])

# Train-Test split
# (For now, I'm taking last week's game to be my test data)
train = final_df[:-10]
test = final_df[-10:].drop(columns='FTR')

train_nolabel = train.drop(columns='FTR')
train_label = train['FTR']

# Machine Learning on data
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(train_nolabel, train_label)
output = clf.predict(test)

# Output
test['FTR'] = np.nan
test['FTR'] = le_res.inverse_transform(output)
test['HomeTeam'] = le_team.inverse_transform(test['HomeTeam'])
test['AwayTeam'] = le_team.inverse_transform(test['AwayTeam'])

print(test.loc[:, {'HomeTeam', 'AwayTeam', 'FTR'}])
