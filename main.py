# 1.Clean the Dataset
# 2.Preprocess the Dataset
# 3.Prepare the Dataset for further analysis

# Team task: Create a credit risk model, which estimates the probability of default  for every personal account
# My task: Take the raw dataset and prepare it for the models the team plan to run

# Loan data we are given, is a sample from a larger dataset that belongs to an affiliate bank from USA, therefore
# -all the values are in Dollars and we need to provide their Euro equivalence
# -Every categorical variable must be quantified
# -We need to change any text columns into numbers
# -For other columns, we only care if they provide positive or negative connotations
# When we're measuring creditworthiness:
# -We need to be extremely risk-averse and distrustful,
# -Missing information suggests foul play.
# -If the information isn't available, we'll assume the worst, where 'worst' depends from column to column


import numpy as np

# Set some filters to how the data will be displayed, however it won't alter the numerical values
np.set_printoptions(suppress=True, linewidth=100, precision=2)

raw_data_np = np.genfromtxt("loan-data.csv",
                            delimiter=';',
                            skip_header=1,
                            autostrip=True)
#print(raw_data_np)


# Checking for Incomplete Data
#print(np.isnan(raw_data_np).sum())


# Creating a filler for all the missing values which will be a number bigger than any other in the dataset
temporary_fill = np.nanmax(raw_data_np) + 1

# Creating a variable that will hold the means for every column
temporary_mean = np.nanmean(raw_data_np, axis=0)
# We get a warning: RuntimeWarning: Mean of empty slice temporary_mean = np.nanmean(raw_data_np, axis=0),
# which means that there're columns with the mean of "nan", therefore, there're no numbers and may consist of strings.

# For convenience we'll create another temporary variable, which will include the min, mean and max of each column
temporary_stats = np.array([np.nanmin(raw_data_np, axis=0),
                            temporary_mean,
                            np.nanmax(raw_data_np, axis=0)])
#print(temporary_stats)



## Splitting the Dataset

# Splitting the Columns
columns_strings = np.argwhere(np.isnan(temporary_mean)).squeeze()
#print(columns_strings)

columns_numeric = np.argwhere(np.isnan(temporary_mean) == False).squeeze()
#print(columns_numeric)

# Now, we'll re-import the dataset, but now we'll use 'usecols' to delimitate the string and numeric columns
load_data_strings = np.genfromtxt("loan-data.csv",
                                  delimiter=';',
                                  skip_header=1,
                                  autostrip=True,
                                  usecols=columns_strings,
                                  dtype=np.str)
#print(load_data_strings)

load_data_numeric = np.genfromtxt("loan-data.csv",
                                  delimiter=';',
                                  skip_header=1,
                                  autostrip=True,
                                  usecols=columns_numeric,
                                  filling_values=temporary_fill)
#print(load_data_numeric)


# The Names of the Columns
header_full = np.genfromtxt("loan-data.csv",
                            delimiter=';',
                            skip_footer=raw_data_np.shape[0],
                            autostrip=True,
                            dtype=np.str)
#print(header_full)

header_strings, header_numeric = header_full[columns_strings], header_full[columns_numeric]
#print(header_strings, header_numeric)

# We'll change the column name of 'issue_d' to 'issue_date' so it would be more descriptive
header_strings[0] = 'issue_date'
#print(load_data_strings)


# Manipulating with each column separately to improve the efficiency, starting with
# Issue Date
load_data_strings[:, 0] = np.chararray.strip(load_data_strings[:, 0], "-15")

# We'll change the months denoted in strings to their numerical representation
months = np.array(['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

for i in range(13):
    load_data_strings[:, 0] = np.where(load_data_strings[:, 0] == months[i],
                                       i,
                                       load_data_strings[:, 0])
#print(np.unique(load_data_strings[:, 0]))

# Loan Status
#print(header_strings)
#print(np.unique(load_data_strings[:, 1]))
# We'll split all possible values from 'Loan Status' into 'good' and 'bad' categories.
# Good: 'Current', 'Fully Paid', 'In Grace Period', 'Issued', 'Late(16-30 days)
# Bad: '', 'Charged off', 'Default', Late(31-120 days)

# We'll assign 1 to all 'good' accounts and 'bad' to all 'bad' accounts
status_bad = np.array(['', 'Charged off', 'Default', 'Late(31-120 days)'])
load_data_strings[:, 1] = np.where(np.isin(load_data_strings[:, 1], status_bad), 0, 1)
#print(np.unique(load_data_strings[:, 1]))

# Term
#print(np.unique(load_data_strings[:, 2]))

# We'll strip the 'months' part from this column and make an addition to the column title in order to be more more clear
load_data_strings[:, 2] = np.chararray.strip(load_data_strings[:, 2], "months")
#print(load_data_strings[:, 2])

header_strings[2] = 'term months'

# To deal with empty spaces, we'll assume the worst, which in this case is 60 (as in 60 months)

load_data_strings[:, 2] = np.where(load_data_strings[:, 2] == '',
                                   '60',
                                   load_data_strings[:, 2])


# Grade and Subgrade

# Filling Sub Grade
# Now we'll check if there're any missing subgrades while having grades. For such cases, we'll add the lowest subgrade
# for the according grade
for i in np.unique(load_data_strings[:, 3])[1:]:
    load_data_strings[:, 4] = np.where((load_data_strings[:, 4] == '') & (load_data_strings[:, 3] == i),
                                       i + "5",
                                       load_data_strings[:, 4])
#print(np.unique(load_data_strings[:, 4 ], return_counts=True))

# For the cases when there's no grade and subgrade, we'll create and assign the worst possible case, which is 'H1'
load_data_strings[:, 4] = np.where((load_data_strings[:, 4] == ''),
                                   'H1',
                                   load_data_strings[:, 4])
#print(np.unique(load_data_strings[:, 4 ], return_counts=True))

# The information carried by 'Grade' is carried now by 'Sub_Grade', therefore, we can delete 'Grade'
load_data_strings = np.delete(load_data_strings, 3, axis=1)
header_strings = np.delete(header_strings, 3)

# Converting the 'Sub-Grade' in numeric format, where 'A1' is 1 and the last one, 'H1' is 36
keys = list(np.unique(load_data_strings[:, 3]))
values = list(range(1, np.unique(load_data_strings[:, 3]).shape[0] + 1))
dict_sub_grade = dict(zip(keys, values))
# Replacing keys with values
for i in np.unique(load_data_strings[:, 3]):
    load_data_strings[:, 3] = np.where(load_data_strings[:, 3] == i,
                                       dict_sub_grade[i],
                                       load_data_strings[:, 3])

# Verification Status
# We'll assign 0 to '' and 'Not Verified', and 1 to 'Verified', and 'Source Verified'
load_data_strings[:, 4] = np.where((load_data_strings[:, 4] == '') | (load_data_strings[:, 4] == 'Not Verified'), 0, 1)

# URL
# So as the links are identical and only the id's are different, we'll strip the identical part
load_data_strings[:, 5] = np.chararray.strip(load_data_strings[:, 5],
                                             'https://www.lendingclub.com/browse/loanDetail.action?loan_id=')
np.array_equal(load_data_strings[:, 5], load_data_strings[:, 0])

# Because the id's int the 'ID' and 'URL' columns are identical, we'll delete the URL column
load_data_strings = np.delete(load_data_strings, 5, axis=1)
header_strings = np.delete(header_strings, 5)


# State Address
# First, we'll alter the column name, to be more descriptive
header_strings[5] = 'state_address'

# In order to get some more useful insights, we'll manipulate the data a bit
states_names, states_count = np.unique(load_data_strings[:, 5], return_counts=True)
states_count_sorted = np.argsort(-states_count)
#print(states_names[states_count_sorted], states_count[states_count_sorted])
# We can see that the most applications come from the wealthy states, and that there're more applications with missing
# or unreported addresses than there are for 45 other states (500 -> '')
# To avoid biasing with variable coefficients, we'll assign 0 to ''
load_data_strings[:, 5] = np.where(load_data_strings[:, 5] == '',
                                   0,
                                   load_data_strings[:, 5])

# In order to find some certain common characteristic, we'll create 4 different arrays, based on geo location
states_west = np.array(['WA', 'OR', 'CA', 'NV', 'ID', 'MT', 'WY', 'UT', 'CO', 'AZ', 'NM', 'HI', 'AK'])
states_south = np.array(['TX', 'OK', 'AR', 'LA', 'MS', 'AL', 'TN', 'KY', 'FL', 'GA', 'SC', 'NC', 'VA', 'WV', 'MD', 
                         'DE', 'DC'])
states_midwest = np.array(['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH'])
states_east = np.array(['PA', 'NY', 'NJ', 'CT', 'MA', 'VT', 'NH', 'ME', 'RI'])

# Assigning numbers to states arrays, west-1, south-2, midwest-3, east-4
load_data_strings[:, 5] = np.where(np.isin(load_data_strings[:, 5], states_west), 1, load_data_strings[:, 5])
load_data_strings[:, 5] = np.where(np.isin(load_data_strings[:, 5], states_south), 2, load_data_strings[:, 5])
load_data_strings[:, 5] = np.where(np.isin(load_data_strings[:, 5], states_midwest), 3, load_data_strings[:, 5])
load_data_strings[:, 5] = np.where(np.isin(load_data_strings[:, 5], states_east), 4, load_data_strings[:, 5])


# Converting the 'load_data_strings' from datatype string to integer
load_data_strings = load_data_strings.astype(np.int)


# Creating a checkpoint function
def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header=checkpoint_header, data=checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return checkpoint_variable


# Checkpoint for all the made changes in the former string dataset
checkpoint_strings = checkpoint("Checkpoint-Strings", header_strings, load_data_strings)



# Manipulating Numeric Columns

# Substitute "Filler" Values with the worst case scenario for each column separately
# ID
np.isin(load_data_numeric[:, 0], temporary_fill)
np.isin(load_data_numeric[:, 0], temporary_fill).sum()

# Funded Amount - the only column for which we need to set the filler value equal to the minimum (worst case scenario)
load_data_numeric[:, 2] = np.where(load_data_numeric[:, 2] == temporary_fill,
                                   temporary_stats[0, columns_numeric[2]],
                                   load_data_numeric[:, 2])

# Loaned Amount, Interest Rate, Total Payment, Installment
for i in [1, 3, 4, 5]:
    load_data_numeric[:, i] = np.where(load_data_numeric[:, i] == temporary_fill,
                                       temporary_stats[2, columns_numeric[i]],
                                       load_data_numeric[:, i])



# Currency Change
# The Exchange Rate - we'll use for this purposes the EUR-USD.csv
EUR_USD = np.genfromtxt("EUR-USD.csv", delimiter=',', autostrip=True, skip_header=1, usecols=3)

# Storing in an array the associate exchange rate for every element of the dataset
exchange_rate = load_data_strings[:, 0]
for i in range(1, 13):
    exchange_rate = np.where(exchange_rate == i,
                             EUR_USD[i-1],
                             exchange_rate)

exchange_rate = np.where(exchange_rate == 0,
                         np.mean(EUR_USD),
                         exchange_rate)

# Need to add to dataset the array
# For that we need to check if they have compatible shapes
# exchange_rate.shape -> (10000,)
# load_data_numeric.shape -> (10000, 6)
# We need to reshape 'exchange_rate'
exchange_rate = np.reshape(exchange_rate, (10000, 1))
load_data_numeric = np.hstack((load_data_numeric, exchange_rate))
# Now we'll add the header to the header_numeric
header_numeric = np.concatenate((header_numeric, np.array(['exchange_rate'])))


# From USD to EUR
columns_dollar = np.array([1, 2, 4, 5])

# Adding Euro version of the columns from columns_dollar which are in Dollar
# We stack horizontally in load_data_numeric new arrays with converted Euro data
for i in columns_dollar:
    load_data_numeric = np.hstack((load_data_numeric,
                                   np.reshape(load_data_numeric[:, i] / load_data_numeric[:, 6], (10000, 1))))


# Expanding the header
# Creating header columns for the new created columns with Euro data
header_additional = np.array([column_name + '_EUR' for column_name in header_numeric[columns_dollar]])
# Adding the new created headers to the 'header_numeric'
header_numeric = np.concatenate((header_numeric, header_additional))
# Changing the header for columns with USD data to be clearer what's what
header_numeric[columns_dollar] = np.array([column_name + '_USD' for column_name in header_numeric[columns_dollar]])

# We will rearrange the columns so that each EUR  column follows its corresponding USD column
columns_index_order = [0, 1, 7, 2, 8, 3, 4, 9, 5, 10, 6]
header_numeric = header_numeric[columns_index_order]
load_data_numeric = load_data_numeric[:, columns_index_order]


# Interest Rate
# The convention dictates using values from 0 to 1 for interest rate, because it makes certain calculations simpler
load_data_numeric[:, 5] = load_data_numeric[:, 5] / 100

# Checkpoint for numeric columns
checkpoint_numeric = checkpoint("Checkpoint-Numeric", header_numeric, load_data_numeric)



# Creating the "Complete" Dataset by stiching the 2 arrays we've got
# For this we need to check if they have compatible shapes
# checkpoint_strings['data'].shape -> (10000, 6)
# checkpoint_numeric['data'].shape -> (10000, 11)
loan_data = np.hstack((checkpoint_numeric['data'], checkpoint_strings['data']))

# Putting together the headers
header_full = np.concatenate((checkpoint_numeric['header'], checkpoint_strings['header']))

# Rearrange the entire dataset according to the values in the first column
loan_data = loan_data[np.argsort(loan_data[:, 0])]

# Attaching all the headers to the Dataset
loan_data = np.vstack((header_full, loan_data))

# Saving the preprocessed Dataset in a csv
np.savetxt('loan-data-preprocessed.csv',
           loan_data,
           fmt="%s",
           delimiter=',')