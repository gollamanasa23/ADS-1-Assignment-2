"""
Module to replace the scipy.stats functions skew(), kurtosis() and bootstrap().
Imports numpy.

@author: napi
"""

""" Module to provide three functions:
    skew to compute the skewness of a distribution
    kurtosis to compute the kurtosis of a distribution
    bootstrap to bootstrap errors of statistical values, like std. dev
"""

import numpy as np


def skew(dist):
    """ Calculates the centralised and normalised skewness of dist. """
    
    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)
    
    # now calculate the skewness
    value = np.sum(((dist-aver) / std)**3) / len(dist-2)
    
    return value


def kurtosis(dist):
    """ Calculates the centralised and normalised excess kurtosis of dist. """
    
    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)
    
    # now calculate the kurtosis
    value = np.sum(((dist-aver) / std)**4) / len(dist-3) - 3.0
    
    return value


def bootstrap(dist, function, confidence_level=0.90, nboot=10000):
    """ Carries out a bootstrap of dist to get the uncertainty of statistical
    function applied to it. Dist can be a numpy array or a pandas dataseries.
    confidence_level specifies the quantile (defaulted to 0.90). E.g 0.90
    means the quantile from 0.05 to 0.95 is evaluated. confidence_level=0.682
    gives the range corresponding to 1 sigma, but evaluated using the 
    corresponding quantiles.
    nboot (default 10000) is the number of bootstraps to be evaluated. 
    Returns the lower and upper quantiles. 
    A call of the form
    low, high = bootstrap(dist, np.mean, confidence_level=0.682)
    will return the lower and upper limits of the 1 sigma range"""
    
    fvalues = np.array([]) # creates an empty array to store function values
    dlen = len(dist)
    for i in range(nboot):
        rand = np.random.choice(dist, dlen, replace=True)
        f = function(rand)
        fvalues = np.append(fvalues, f)
        
    # lower and upper quantiles
    qlow = 0.5 - confidence_level/2.0
    qhigh = 0.5 + confidence_level/2.0

    low = np.quantile(fvalues, qlow)
    high = np.quantile(fvalues, qhigh)
    
    return low, high


# checks whether module is imported or run directly.
# code is not executed if imported
if __name__ == "__main__":
    
    dist = np.random.normal(4.0, 3.0, 10000)
    
    print("skewness =", np.round(skew(dist), 6))
    print("kurtosis =", np.round(kurtosis(dist), 6))
    
    print()
    # Call the boostrap routine with statistical functions
    low, high = bootstrap(dist, np.mean, confidence_level=0.682)
    sigma = 0.5 * (high - low)
    print("average = ", np.round(np.mean(dist), 4), "+/-", np.round(sigma, 4))
    
    low, high = bootstrap(dist, np.std, confidence_level=0.682)
    sigma = 0.5 * (high - low)
    print("std. dev = ", np.round(np.std(dist), 4), "+/-", np.round(sigma, 4))
    
    low, high = bootstrap(dist, skew, confidence_level=0.682)
    sigma = 0.5 * (high - low)
    print("skewness = ", np.round(skew(dist), 4), "+/-", np.round(sigma, 4))
    
    low, high = bootstrap(dist, kurtosis, confidence_level=0.682)
    sigma = 0.5 * (high - low)
    print("kurtosis = ", np.round(kurtosis(dist), 4), "+/-", 
          np.round(sigma, 4))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
# Assuming your data is in a CSV file named 'your_data_file.csv'
data_file = 'API_19_DS2_en_csv_v2_6183479.csv'

# Read the dataset into a pandas DataFrame
df = pd.read_csv(data_file)

# Display the first few rows of the DataFrame to identify column names
print(df.head())
df_cleaned = df.dropna()

print(df_cleaned.head())

# Provided numerical values
years = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
value1 = [45.606, 45.24, 44.875, 44.511, 44.147, 43.783, 43.421, 43.059, 42.94, 42.957, 42.99, 43.041, 43.108, 43.192, 43.293, 43.411, 43.546, 43.697, 43.866, 44.052]
value2 = [42277, 42317, 42399, 42555, 42729, 42906, 43079, 43206, 43493, 43864, 44228, 44588, 44943, 45297, 45648, 45999, 46351, 46574, 46734, 46891]

# Create a DataFrame
data = {'Year': years, 'Value1': value1, 'Value2': value2}
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

import matplotlib.pyplot as plt

# Assuming you have already loaded and organized your data into the DataFrame 'df'

# Line plot for Value1 over the years
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Value1'], marker='o', label='Value1')
plt.title('Line Plot of Value1 over the Years')
plt.xlabel('Year')
plt.ylabel('Value1')
plt.legend()
plt.show()

# Bar plot for Value2 over the years
plt.figure(figsize=(10, 6))
plt.bar(df['Year'], df['Value2'], color='skyblue')
plt.title('Bar Plot of Value2 over the Years')
plt.xlabel('Year')
plt.ylabel('Value2')
plt.show()

# Scatter plot for Value1 and Value2
plt.figure(figsize=(10, 6))
plt.scatter(df['Value1'], df['Value2'], color='green', alpha=0.7)
plt.title('Scatter Plot between Value1 and Value2')
plt.xlabel('Value1')
plt.ylabel('Value2')
plt.show()

import seaborn as sns

# Example of using seaborn for a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Year', y='Value1', data=df)
plt.title('Box Plot of Value1 over the Years')
plt.xlabel('Year')
plt.ylabel('Value1')
plt.show()



# Assuming you have already loaded and organized your data into the DataFrame 'df'

# Line plot for Value1 over the years with a trendline
plt.figure(figsize=(12, 6))
sns.regplot(x='Year', y='Value1', data=df, ci=None, line_kws={"color": "red"})
plt.title('Line Plot of Value1 over the Years with Trendline')
plt.xlabel('Year')
plt.ylabel('Value1')
plt.show()

# Area plot for Value1 and Value2
plt.figure(figsize=(12, 6))
plt.fill_between(df['Year'], df['Value1'], label='Value1', alpha=0.5)
plt.fill_between(df['Year'], df['Value2'], label='Value2', alpha=0.5)
plt.title('Area Plot of Value1 and Value2 over the Years')
plt.xlabel('Year')
plt.ylabel('Values')
plt.legend()
plt.show()

# Violin plot for Value2
plt.figure(figsize=(10, 6))
sns.violinplot(x=df['Year'], y=df['Value2'], inner='quartile')
plt.title('Violin Plot of Value2 over the Years')
plt.xlabel('Year')
plt.ylabel('Value2')
plt.show()

# Pair plot for all numerical columns
sns.pairplot(df[['Value1', 'Value2']], height=2)
plt.suptitle('Pair Plot of Value1 and Value2', y=1.02)
plt.show()

# Heatmap for correlation between numerical columns
correlation_matrix = df[['Value1', 'Value2']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()


