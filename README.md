# load_data_numpy

1.Clean the Dataset

2.Preprocess the Dataset

3.Prepare the Dataset for further analysis


## Team task: Create a credit risk model, which estimates the probability of default  for every personal account
## My task: Take the raw dataset and prepare it for the models the team plan to run.


Loan data we are given, is a sample from a larger dataset that belongs to an affiliate bank from USA, therefore:

-all the values are in Dollars and we need to provide their Euro equivalence

-Every categorical variable must be quantified

-We need to change any text columns into numbers

-For other columns, we only care if they provide positive or negative connotations


## When we're measuring creditworthiness:

-We need to be extremely risk-averse and distrustful,

-Missing information suggests foul play.

-If the information isn't available, we'll assume the worst, where 'worst' depends from column to column
