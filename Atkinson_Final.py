#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 19:16:28 2023

@author: milesatkinson
"""

# =============================================================================
# Import packages
# =============================================================================
## Base packages
import pandas as pd
import numpy as np
## Packages for APIs
import json
import requests
import os
## Packages for visualization
import matplotlib.pyplot as plt
## Packages for statistics
from scipy import stats
from scipy.stats import ttest_ind, t
import statsmodels.api as sm
import statsmodels.formula.api as smf



# =============================================================================
# Import datasets 
# =============================================================================

census = pd.read_csv('Clean Census Data.csv')
atlanta = pd.read_excel('STANDARDS SALES 2018.xlsx')

## Standardize column names
census.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
atlanta.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)


# =============================================================================
# API attempt for Census
# =============================================================================


# ##Census API Key: 0fd6c2a35a4eaff9fcb752d084a56c7ae103bee5
# url = 'https://api.census.gov/data/2018/acs/acs5'

# ## set parameters
# params = {
#     'get': 'NAME,B01001_001E,B02001_001E,B02001_002E,B02001_003E,B02001_007E,B02001_008E',
#     'for': 'tract:*', 
#     'in': 'county:121',
#     'key': '0fd6c2a35a4eaff9fcb752d084a56c7ae103bee5'
#     }
# census_api = requests.get(url, params=params)

# ## check successful call
# census_api.status_code

# ## Convert JSON response into Dataframe
# census_json = census_api.json()   #JSON
# census_df=pd.DataFrame(census_json[1:], columns=census_json[0])


# =============================================================================
# Atlanta Data Wrangling
# ============================================================================= 

# Engineer a few new values

## Replace sales values that equal $0 with $1
atlanta['sales_price'].replace(0,1, inplace=True)

## Calculates the "real assessment ratio" based on sales price compared to fair market value
atlanta['real_assess_ratio'] = atlanta['assessed_value'] / atlanta['sales_price']

## Creates boolean indicator for overassessment
atlanta['overassessed'] = atlanta['real_assess_ratio'] > 0.41

# =============================================================================

## Use Census Geocoder to add geographic data to the Atlanta dataset
## https://geocoding.geo.census.gov/geocoder/geographies/addressbatch?form

## subset atlanta to get useful address details for Census Geocoder
atlanta_addresses = atlanta.loc[:, ['parid', 'adrpre', 'adrno', 'adrdir',
                                         'adrstr', 'adrsuf', 'adrsuf2', 
                                         'cityname', 'real_assess_ratio',
                                         'overassessed']]

atlanta_addresses['state'] = "Georgia"    # missing state name
atlanta_addresses['adrno'] = atlanta_addresses['adrno'].apply(str)  # make address number a string
atlanta_addresses['address'] = atlanta_addresses['adrno'] + ' ' + atlanta_addresses['adrstr'] + ' ' + atlanta_addresses['adrsuf']
atladd = atlanta_addresses.loc[:, ['address', 'cityname', 'state', 'real_assess_ratio', 'overassessed']]

atladd['zip_code'] = ''         # CensusGeocoder needs a particular format with zip code

## Create two sets of data for geocoding.

# Sample 9999 random properties
atladd_10 = atladd.sample(n=9999, random_state=42)
atladd_10.drop(['real_assess_ratio', 'overassessed'], axis=1)
atladd_10.dropna(subset=['address'], inplace=True)

# Select properties that are overassessed
atladd_oa = atladd.loc[atladd['overassessed'] == True]
atladd_oa = atladd_oa.drop(['real_assess_ratio','overassessed'], axis=1)
atladd_oa.dropna(subset=['address'], inplace=True)



## Write to csv
atladd_10.to_csv('atladd_10.csv', header = False)
atladd_oa.to_csv('atladd_oa.csv', header = False)

# =============================================================================
# Census GEOCODER API attempt
# geo_url = 'https://geocoding.geo.census.gov/geocoder/geographies/addressbatch'
# geo_params = {'benchmark': 'Public_AR_Current',
#               'vintage': 'Census2020_Current',
#               'format': 'text/csv'}

# with open('atladd_oa.csv', 'rb') as f:
#     atl_geo = requests.post(geo_url, data=f, params=geo_params, headers={'Content-Type': 'text/csv'})
    
# ## Census API request is returning a Response 500 error.

# =============================================================================

## In the meantime, I used Census Geocoder tool for a batch of Atlanta addresses
# https://geocoding.geo.census.gov/geocoder/geographies/addressbatch?form

atl_geobatch_10 = pd.read_csv('GeocodeResults_10.csv')  
atl_geobatch_oa = pd.read_csv('GeocodeResults_oa.csv')  

dfs = [atl_geobatch_oa, atl_geobatch_10]

# loop through the list and perform the data wrangling each dataframe
for df in dfs:
    # Add column headers
    df.columns = ['index', 'Address', 'TIGER Indicator', 'TIGER Match Type',
                  'TIGER Output Address', 'Interpolated LAT-LONG', 'TIGERLINE ID',
                  'State Code', 'County Code', 'Tract Code', 'Block Code', '?']
    df.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True) # standardize
    # Merge the Census Geocoder and Atlanta Address datasets
    new = df["address"].str.split(",", n = 2, expand = True)
    df['address'] = new[0]
    df['city'] = new[1]
    df['state'] = new[2]
    df.dropna(subset = ['tiger_match_type'], inplace=True)



# ## Left join to get parcel id into the dataset
atl_oa = pd.merge(atl_geobatch_oa, atlanta_addresses[['parid','overassessed']],
                  left_on= 'index', right_index=True, how='left')

atl_10  = pd.merge(atl_geobatch_10, atlanta_addresses[['parid', 'overassessed']],
                    left_on= 'index', right_index=True, how='left')

    
## Merge the Geographic information with the relevant columns from overassesed dataset
atl_oa = pd.merge(atl_oa, atlanta[['sales_price', 'fair_market_value', 
                              'assessed_value', 'real_assess_ratio']],
                left_on='index', right_index=True, how='left')

atl_10 = pd.merge(atl_10, atlanta[['sales_price', 'fair_market_value', 
                              'assessed_value', 'real_assess_ratio']],
                left_on = 'index', right_index=True, how='left')

## Drop unneccessary columns, duplicates, NaN values
atl_oa = atl_oa.drop(['tiger_indicator', 'tiger_match_type', '?'], axis=1)
atl_10 = atl_10.drop(['tiger_indicator', 'tiger_match_type', '?'], axis=1)




## Add decimal places to Block Code for merge with Census data
atl_oa['census_tract'] = atl_oa['block_code']/100
atl_10['census_tract'] = atl_10['block_code']/100


# =============================================================================
# Census Data Wrangling
# =============================================================================

## Put the census tract data into the same format as atl
new1 = census['census_tract'].str.split(' ', n=3, expand=True)
census['census_tract'] = new1[3]

## Convert the string into a float
census['census_tract'] = census['census_tract'].astype(float)

## The census data is more granular than my Atlanta data, so need to count total
## population across the census blocks within the census tracts

## Sum all the values within census tracts
grouped_census = census.groupby('census_tract', as_index=False).sum()

## Engineer a few variables
grouped_census['percent_black'] = grouped_census['black_alone'] / grouped_census['total']
grouped_census['percent_hispanic'] = grouped_census['hispanic_or_latino'] / grouped_census['total']
grouped_census['percent_nonwhite'] = (grouped_census['total'] - grouped_census['white_alone']) / grouped_census['total']

## Test to make sure the numbers match up 
mask = grouped_census['census_tract'].isin(atl_oa['census_tract'])
mask1 = grouped_census['census_tract'].isin(atl_10['census_tract'])

pd.Series(mask).value_counts()
pd.Series(mask1).value_counts()

# =============================================================================
# Merge the datasets
# =============================================================================

final_oa = pd.merge(atl_oa, grouped_census[['census_tract', 'total', 'hispanic_or_latino', 'white_alone',
                              'black_alone', 'percent_black', 'percent_hispanic', 'percent_nonwhite']],
                 on = 'census_tract', how = 'left')

# drop NaN value for future plotting
final_oa.dropna(subset = ['total'], inplace=True)

final_10 = pd.merge(atl_10, grouped_census[['census_tract', 'total', 'hispanic_or_latino', 'white_alone',
                              'black_alone', 'percent_black', 'percent_hispanic', 'percent_nonwhite']],
                 on = 'census_tract', how = 'left')


# =============================================================================
# Summary Statistics
# =============================================================================

# Basic information
dafa = [final_oa, final_10]

for df in dafa:
    print(df.shape)
    print(df.columns)
    # create a table of summary statistics
    print(df[['sales_price', 'fair_market_value', 'real_assess_ratio',
                      'total', 'percent_black', 'percent_nonwhite']].describe())
# =============================================================================

pretty_table_oa = final_oa[['sales_price', 'fair_market_value', 'real_assess_ratio',
                  'total', 'percent_black', 'percent_nonwhite']].describe()


# =============================================================================
#  Build graphics
# =============================================================================

# =============================================================================
# Boxplots

## Fair Market Value and Sales Price
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

v = final_oa[['fair_market_value', 'sales_price']]
rar = final_oa['real_assess_ratio']

plt.ylabel('Price')
plt.title('Boxplot of Fair Market Value and Sales Price')
plt.boxplot(v, notch=True, vert=False, patch_artist=True, 
            boxprops=dict(facecolor='#3399ff'), medianprops=dict(color='white'), 
            capprops=dict(color='black'), whiskerprops=dict(color='black'), 
            flierprops=dict(marker='o', markerfacecolor='red', markersize=5), 
            showmeans=True, meanline=True, meanprops=dict(color='green'))
plt.show()   

plt.boxplot(rar)
plt.ylabel('Real Assessment to Sale Price Ratio')
plt.title('Boxplot of the Ratio of Assessment Value to Actual Sales Price')
plt.show()      


### The above plots demonstrate the presence of a large outliers, which I will remove
v1 = final_oa[(final_oa['fair_market_value'] < 500000) & (final_oa['sales_price'] < 500000)]
v1 = v1[['fair_market_value', 'sales_price']]

plt.boxplot(v1, notch=True, vert=False, patch_artist=True, 
            boxprops=dict(facecolor='#3399ff'), medianprops=dict(color='white'), 
            capprops=dict(color='black'), whiskerprops=dict(color='black'), 
            flierprops=dict(marker='o', markerfacecolor='red', markersize=5), 
            showmeans=True, meanline=True, meanprops=dict(color='green'))
plt.title('Boxplot of Fair Market Value and Sales Price (Outliers Removed)')
plt.show()   

# =============================================================================
# Histogram

# Generate some data
ab = final_oa.loc[final_oa['sales_price'] < 500000, ['sales_price']]
ab['fair_market_value'] = final_oa.loc[final_oa['fair_market_value'] < 500000, ['fair_market_value']]

cd = final_10['percent_black']

ef = final_oa.loc[final_oa['real_assess_ratio'] < 1, ['real_assess_ratio']]
gh = atlanta.loc[atlanta['real_assess_ratio'] < 1, ['real_assess_ratio']]


# Create subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)


# Plot histograms
axs[0, 0].hist(ab, bins=20, color=['r', 'b'], alpha=0.8)
axs[0, 0].set_title('Sales Price and Fair Market Value\nfor Over-assessed Properties')
axs[0, 0].set_xlabel('Sales Price and FMV')
axs[0, 0].set_ylabel('Frequency')

axs[0, 1].hist(cd, bins=20, color='g')
axs[0, 1].set_title('Percentage Black Population\nin Fulton County Census Tracts')
axs[0, 1].set_xlabel('Percentage Black Population')
axs[0, 1].set_ylabel('Frequency')

axs[1, 0].hist(ef, bins=30, color='r')
axs[1, 0].set_title('"Real Assessment Ratio" for\nover-assessed properties')
axs[1, 0].set_xlabel('Real Assessment Ratio')
axs[1, 0].set_ylabel('Frequency')

axs[1, 1].hist(gh, bins=30, color='c', alpha=0.5)
axs[1, 1].set_title('"Real Assessment Ratio"\nfor all properties in Atlanta')
axs[1, 1].set_xlabel('Real Assessment Ratio')
axs[1, 1].set_ylabel('Frequency')

# Add a super title
fig.suptitle('Histograms of Dataset variables')

# Display the plot
plt.show()

# FMV in white areas versus minority areas
wfmv = final_oa.loc[(final_oa['percent_nonwhite'] < 0.2) & (final_oa['fair_market_value'] < 500000), 'fair_market_value']
mfmv = final_oa.loc[(final_oa['percent_nonwhite'] > 0.8) & (final_oa['fair_market_value'] < 500000), 'fair_market_value']
bfmv = final_oa.loc[(final_oa['percent_black'] > 0.8) & (final_oa['fair_market_value'] < 500000), 'fair_market_value']

# Set up the subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot the histograms
axs[0].hist(wfmv, bins=8, range=(0, 500000))
axs[0].set_title('Percent non-white < 0.2')
axs[0].set_xlabel('Fair market value')
axs[0].set_ylabel('Frequency')

axs[1].hist(mfmv, bins=10, range=(0, 500000))
axs[1].set_title('Percent non-white > 0.8')
axs[1].set_xlabel('Fair market value')
axs[1].set_ylabel('Frequency')

axs[2].hist(bfmv, bins=10, range=(0, 500000))
axs[2].set_title('Percent black > 0.8')
axs[2].set_xlabel('Fair market value')
axs[2].set_ylabel('Frequency')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()   

# =============================================================================
# Scatterplots

## RAR by Percent Black and Percent Non-White
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot 1
x = final_oa.loc[final_oa['real_assess_ratio'] < 1, ['percent_black']]  
y = final_oa.loc[final_oa['real_assess_ratio'] < 1, ['real_assess_ratio']]
xr = np.ravel(x)
yr = np.ravel(y)
im1 = ax1.scatter(xr, yr, c=xr, cmap='plasma')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([.38, 1.5])
ax1.set_xlabel('Population Percent Black')
ax1.set_ylabel('Real Assessment Ratio')
ax1.set_title('Real Assessment Ratio by Percentage Black')
z = np.polyfit(xr, yr, 1)
p = np.poly1d(z)
ax1.plot(xr, p(xr))

# Plot 2
x1 = final_oa.loc[final_oa['real_assess_ratio'] < 1, ['percent_nonwhite']] 
x1r = np.ravel(x1)
im2 = ax2.scatter(x1r, yr, c=x1r, cmap='plasma')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([.38, 1.5])
ax2.set_xlabel('Population Percent Non-White')
ax2.set_ylabel('Real Assessment Ratio')
ax2.set_title('Real Assessment Ratio by Percentage Non-White')
z = np.polyfit(x1r, yr, 1)
p = np.poly1d(z)
ax2.plot(x1r, p(x1r))

# Add colorbar
cbar = fig.colorbar(im1, ax=[ax1, ax2])
cbar.set_label('Percent Black', labelpad=8)

plt.show()



# =============================================================================
# Spearman Rank Correlation Coefficient
# =============================================================================

# Are homes more heavily overassessed in minority areas?

rar = final_oa.loc[final_oa['real_assess_ratio'] < 1.0, ['real_assess_ratio']].to_numpy()

## Create arrays for percent black and percent nonwhite
pnw = final_oa.loc[final_oa['real_assess_ratio'] < 1.0, ['percent_nonwhite']].to_numpy()
pb = final_oa.loc[final_oa['real_assess_ratio'] < 1.0, ['percent_black']].to_numpy()

## Spearman's Rank Correlation Coefficient
corr_coef_pnw, p_value_pnw = stats.spearmanr(pnw, rar)
corr_coef_pb, p_value_pb = stats.spearmanr(pb, rar)

# print the results
print(f"Spearman rank correlation coefficient: {corr_coef_pnw:.3f}")
print(f"P-value: {p_value_pnw:.5f}")
print(f"Spearman rank correlation coefficient: {corr_coef_pb:.3f}")
print(f"P-value: {p_value_pb:.5f}")

# =============================================================================
# T-test
# =============================================================================

# Create a dataframe with  continuous variable and a True/False
t_test = final_10[['percent_black', 'percent_nonwhite', 'overassessed']]

# Divide into two
t_true = t_test.loc[t_test['overassessed'] == True, 'percent_black']
t_false = t_test.loc[t_test['overassessed'] == False, 'percent_black'].dropna()


# perform the t-test
t_stat, p_val = ttest_ind(t_true, t_false)

print("T-statistic: ", t_stat)
print("P-value: ", p_val)

## T-statistic is 23.98102346480484, DF is 8504
df = 8504
alpha = 0.05
cv = t.ppf(1 - alpha/2, df)

print("Critical value:", cv)
## Critical value: 1.9602429830311303


print("Percent black population for over-assessed homes: {:.2f}%".format(round(t_true.mean()*100, 2)))
print("Percent black population for properly assessed homes: {:.2f}%".format(round(t_false.mean()*100, 2)))


# =============================================================================
# Linear Regression
# =============================================================================

# Predicted value of the "real assessment ratio" based on
# percent of the population that is non-white or black
model_pnw = sm.OLS(rar, sm.add_constant(pnw)).fit()
model_pb = sm.OLS(rar, sm.add_constant(pb)).fit()


# Print the summary statistics of the models
# For "percent non-white"
model_pnw.summary()
pnw_predict = model_pnw.predict(sm.add_constant(pnw))
# For "percent black"
print(model_pb.summary())
pb_predict = model_pb.predict(sm.add_constant(pb))

# Create a scatter plot with the actual and predicted values
plt.scatter(pb, rar, label='Actual')
plt.scatter(pb, pb_predict, label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Model')
plt.legend()
plt.show()


# =============================================================================
# Logistic Regression
# =============================================================================
logdata = final_10.dropna(subset = ['percent_black'])
logx = logdata['percent_black']
logy = logdata['overassessed']

logx = sm.add_constant(logx)  # add an intercept term
logit_model = sm.Logit(logy, logx)
result_log = logit_model.fit()

## Summary
print(result_log.summary())

## Odds ratio
np.exp(result_log.params)

# Generate predictions from the model
pb_log_predict = result_log.predict(sm.add_constant(logx))

logx2 = logdata['percent_black']


# Create a scatter plot with the actual and predicted values
plt.scatter(logx2, logy, label='Actual')
plt.scatter(logx2, pb_log_predict, label='Predicted')
plt.xlabel('Percent Black')
plt.ylabel('True (1) / False (0)')
plt.title('Log Model')
plt.legend()
plt.show()

# =============================================================================
# 
# =============================================================================





