from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as numpy
import pandas as pandas
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data_frame = pandas.read_csv("train.csv")
test = pandas.read_csv("test.csv")
loan_approval = data_frame['Loan_Status'].value_counts()['Y']

# Function to output percentage row wise in a cross table


def percentageConvert(ser):
    return ser/float(ser[-1])


# Replace missing value of Self_Employed with more frequent category
data_frame['Self_Employed'].fillna('No', inplace=True)

# Add both ApplicantIncome and CoapplicantIncome to TotalIncome
data_frame['TotalIncome'] = data_frame['ApplicantIncome'] + \
    data_frame['CoapplicantIncome']

# Looking at the distribtion of TotalIncome
data_frame['LoanAmount'].hist(bins=20)
# Perform log transformation of TotalIncome to make it closer to normal
data_frame['LoanAmount_log'] = numpy.log(data_frame['LoanAmount'])

# Looking at the distribtion of TotalIncome_log
data_frame['LoanAmount_log'].hist(bins=20)
# Impute missing values for Gender
data_frame['Gender'].fillna(data_frame['Gender'].mode()[0], inplace=True)

# Impute missing values for Married
data_frame['Married'].fillna(data_frame['Married'].mode()[0], inplace=True)

# Impute missing values for Dependents
data_frame['Dependents'].fillna(
    data_frame['Dependents'].mode()[0], inplace=True)

# Impute missing values for Credit_History
data_frame['Credit_History'].fillna(
    data_frame['Credit_History'].mode()[0], inplace=True)

# Convert all non-numeric values to number
cat = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Credit_History', 'Property_Area']

for var in cat:
    le = preprocessing.LabelEncoder()
    data_frame[var] = le.fit_transform(data_frame[var].astype('str'))
data_frame.dtypes


# Import models from scikit learn module:

# Generic function for making a classification model and accessing performance:


def classification_model(model, data, predictors, outcome):
    # Fit the model:
    model.fit(data[predictors], data[outcome])

    # Make predictions on training set:
    predictions = model.predict(data[predictors])

    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(n_splits=5)
    error = []
    for train, test in kf.split(data[predictors]):
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record error from each cross-validation run
        error.append(model.score(
            data[predictors].iloc[test, :], data[outcome].iloc[test]))

    print("Cross-Validation Score : %s" % "{0:.3%}".format(numpy.mean(error)))

    # Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors], data[outcome])
# Combining both train and test dataset


# Create a flag for Train and Test Data set
data_frame['Type'] = 'Train'
test['Type'] = 'Test'
fullData = pandas.concat([data_frame, test], axis=0)

# Look at the available missing values in the dataset
fullData.isnull().sum()
# Identify categorical and continuous variables
ID_col = ['Loan_ID']
target_col = ["Loan_Status"]
cat_cols = ['Credit_History', 'Dependents', 'Gender',
            'Married', 'Education', 'Property_Area', 'Self_Employed']
# Imputing Missing values with mean for continuous variable
fullData['LoanAmount'].fillna(fullData['LoanAmount'].mean(), inplace=True)
fullData['LoanAmount_log'].fillna(
    fullData['LoanAmount_log'].mean(), inplace=True)
fullData['Loan_Amount_Term'].fillna(
    fullData['Loan_Amount_Term'].mean(), inplace=True)
fullData['ApplicantIncome'].fillna(
    fullData['ApplicantIncome'].mean(), inplace=True)
fullData['CoapplicantIncome'].fillna(
    fullData['CoapplicantIncome'].mean(), inplace=True)

# Imputing Missing values with mode for categorical variables
fullData['Gender'].fillna(fullData['Gender'].mode()[0], inplace=True)
fullData['Married'].fillna(fullData['Married'].mode()[0], inplace=True)
fullData['Dependents'].fillna(fullData['Dependents'].mode()[0], inplace=True)
fullData['Loan_Amount_Term'].fillna(
    fullData['Loan_Amount_Term'].mode()[0], inplace=True)
fullData['Credit_History'].fillna(
    fullData['Credit_History'].mode()[0], inplace=True)
# Create a new column as Total Income

fullData['TotalIncome'] = fullData['ApplicantIncome'] + \
    fullData['CoapplicantIncome']

fullData['TotalIncome_log'] = numpy.log(fullData['TotalIncome'])

# Histogram for Total Income
fullData['TotalIncome_log'].hist(bins=20)
# create label encoders for categorical features
for var in cat_cols:
    number = LabelEncoder()
    fullData[var] = number.fit_transform(fullData[var].astype('str'))

train_modified = fullData[fullData['Type'] == 'Train']
test_modified = fullData[fullData['Type'] == 'Test']
train_modified["Loan_Status"] = number.fit_transform(
    train_modified["Loan_Status"].astype('str'))


predictors_Logistic = ['Credit_History', 'Education', 'Gender']

x_train = train_modified[list(predictors_Logistic)].values
y_train = train_modified["Loan_Status"].values

x_test = test_modified[list(predictors_Logistic)].values
# Create logistic regression object
model = LogisticRegression()

# Train the model using the training sets
model.fit(x_train, y_train)

# Predict Output
predicted = model.predict(x_test)

# Reverse encoding for predicted outcome
predicted = number.inverse_transform(predicted)

# Store it to test dataset
test_modified['Loan_Status'] = predicted

outcome_var = 'Loan_Status'

classification_model(model, data_frame, predictors_Logistic, outcome_var)

test_modified.to_csv("Logistic_Prediction.csv",
                     columns=['Loan_ID', 'Loan_Status'])
