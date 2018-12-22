# epl_data_analysis
Small machine learning project for analysing EPL match data to predict the outcomes (home win, away win, or draw) of a match.
Raw data can be found at:
http://www.football-data.co.uk/data.php

## Process
We take English Premier League data for matches from 2010 to present (collected from football-data.co.uk) and analyze it using multiple classification algorithms. 

## Data Cleaning
In order to deal with missing columns, I did an inner join of all seasons of data (from 2010 till present). This ensures that I don't have columns with data not present or filled with nan values. 
I aggregate data values by replacing data with data averages over matches (i.e. shots on target for a game is replaced by average shots on target for the team for it's past matches). This is done because otherwise we would train on data that happens during the match, which would not be available pre-match. 
Finally, I used pandas' dropna function on dataframes to get rid of any columns that have nan values in them. Due to our process, we will get nan columns (such as the first ten matches for which there is no available data to make averages from) which I decided to remove as something like 20 matches gone from 3000 won't make a huge difference.
Removed all the columns that were not needed (such as the columns containing the post-match data)
Finally, I split the data into training and testing data so that I could verify the accuracy of my models.

## Label Encoding
Most Classification Algorithms are unable to deal with non-numerical features, and thus I had to convert HomeTeam, AwayTeam, and FTR into integers. I used sklearn's LabelEncoder classes to deal with this.

## Machine Learning
Using scikit-learn's extensive library for Classification Algorithms, there is the option to choose between five separate algorithms in the code. I've been working with these five mostly, and for the EPL data I seem to get the best result with random forest. I get consistent accuracies of around 55-58% which seems to be the best I can get out of the models on this basic level. 
As of now, I've been tuning hyperparameters for random forest, trying to find the best combination.

Contact me: ayushlall@g.ucla.edu
