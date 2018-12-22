# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:36:53 2018

@author: hadoop
"""

import pandas as pd
import numpy as np

# EPL Data analysis
# Taking data from www.football-data.co.uk


# Flag for if we have the data already stored somewhere
HAVE_DATA = True;

RANDOM_STATE = 42;

# Size of set taken for testing
TEST_SIZE = 0.3;

# Algorithm options: 'random_forest', 'decision_tree', 
# 'logistic_regression', 'gaussian_nb', 'knn', 'sgd'
ALG = 'random_forest'

folder_path = 'path/to/data/here'

# Aggregates the column values using time (need better commenting for this function)
# aggregator: A function that does the required data processing for a given
# type of information. In the EPL data, we cannot use data points such as FTHG, FTAG
# and so on as these are values observed DURING the game. For match predictions to 
# be made before the game, using these values would make all our predictions rubbish.
# So, the way to deal with this is to find the average values of match data before
# the match - which is what this function does. 

# Note: This function needs optimization will work on it asap
def aggregator (df, home, home_column, away_column, output_column):
    for index, row in df.iterrows():
        hometeam = row[home]
        df_slice = df[df['Date'] < row['Date']]
        if (len(df_slice) == 0): 
            continue
        sum_series = df_slice.loc[(df_slice['HomeTeam'] == hometeam)][home_column].append(df_slice.loc[(df['AwayTeam'] == hometeam)][away_column])
        df[output_column].iat[index] = sum_series.mean()
        

if (not HAVE_DATA):
   
    # Getting data
    df_2018 = pd.read_csv(folder_path + '2018-2019.csv')
    df_2017 = pd.read_csv(folder_path + '2017-2018.csv')
    df_2016 = pd.read_csv(folder_path + '2016-2017.csv')
    df_2015 = pd.read_csv(folder_path + '2015-2016.csv')
    df_2014 = pd.read_csv(folder_path + '2014-2015.csv')
    df_2013 = pd.read_csv(folder_path + '2013-2014.csv')
    df_2012 = pd.read_csv(folder_path + '2012-2013.csv')
    df_2011 = pd.read_csv(folder_path + '2011-2012.csv')
    df_2010 = pd.read_csv(folder_path + '2010-2011.csv')
    # Concatenating the data
    df = pd.concat([df_2018, df_2017, df_2016, df_2015, df_2014, df_2013, df_2012, df_2011, df_2010] , axis=0, join='inner', ignore_index=True)
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
    
    
    aggregator(df,'HomeTeam','FTHG','FTAG','FTHG_AGG')
    aggregator(df,'AwayTeam','FTHG','FTAG','FTAG_AGG')
    aggregator(df,'HomeTeam','HTHG','HTAG','HTHG_AGG')
    aggregator(df,'AwayTeam','HTHG','HTAG','HTAG_AGG')
    aggregator(df,'HomeTeam','HS','AS','HS_AGG')
    aggregator(df,'AwayTeam','HS','AS','AS_AGG')
    aggregator(df,'HomeTeam','HST','AST','HST_AGG')
    aggregator(df,'AwayTeam','HST','AST','AST_AGG')
    aggregator(df,'HomeTeam','HF','AF','HF_AGG')
    aggregator(df,'AwayTeam','HF','AF','AF_AGG')
    aggregator(df,'HomeTeam','HC','AC','HC_AGG')
    aggregator(df,'AwayTeam','HC','AC','AC_AGG')
    aggregator(df,'HomeTeam','HY','AY','HY_AGG')
    aggregator(df,'AwayTeam','HY','AY','AY_AGG')
    aggregator(df,'HomeTeam','HR','AR','HR_AGG')
    aggregator(df,'AwayTeam','HR','AR','AR_AGG')
    
    # Data cleaning: dropping post-game data
    df = df.sort_values(by=['Date'])
    final_df = df.drop(columns=['Div','Date','Referee','HTR','FTHG','FTAG','HTHG','HTAG','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR'])
    final_df.to_csv(folder_path + '2010_final_df.csv')
    
else:
    final_df = pd.read_csv(folder_path + '2010_final_df.csv')
    
    
# Data Cleaning: Removing nan values
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

from sklearn.model_selection import train_test_split
train, test = train_test_split(final_df, test_size=0.3, random_state=RANDOM_STATE)


#train = final_df[:-TEST_SIZE]
#test = final_df[-TEST_SIZE:].drop(columns='FTR')
test_label = test['FTR']
test = test.drop(columns='FTR')

train_nolabel = train.drop(columns='FTR')
train_label = train['FTR']

# Different Machine Learning Algorithms

if ALG is 'logistic_regression':
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(train_nolabel, train_label)
    output = clf.predict(test)
    
if ALG is 'gaussian_nb':
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(train_nolabel, train_label)
    output = clf.predict(test)

if ALG is 'random_forest':
    
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, max_features='auto', max_leaf_nodes=100, min_samples_leaf=110, min_samples_split=5, bootstrap=True)    
    clf.fit(train_nolabel, train_label) 
    output = clf.predict(test)
    
if ALG is 'decision_tree':
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(train_nolabel, train_label)
    output = clf.predict(test)
    
if ALG is 'knn':
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
    clf.fit(train_nolabel, train_label)
    output = clf.predict(test)

if ALG is 'sgd':
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss='perceptron',penalty='elasticnet')
    clf.fit(train_nolabel, train_label)
    output = clf.predict(test)


# Output
    
#inverse_transform = {1 : 'H', 0 : 'D', 2 : 'A'}
#test['FTR'] = np.nan
#test['FTR'] = le_res.inverse_transform(output)
#test['HomeTeam'] = le_team.inverse_transform(test['HomeTeam'])
#test['AwayTeam'] = le_team.inverse_transform(test['AwayTeam'])

#print(test.loc[:,['HomeTeam', 'AwayTeam', 'FTR']][::-1])
    
from sklearn.metrics import confusion_matrix
score = (test_label == output).sum() / len(output)
print(ALG)
print('random_state: ', RANDOM_STATE)
print("raw_score: ", score)
print("percentage_confusion_matrix: ")
cm = confusion_matrix(test_label, output)
print(cm / cm.sum())


