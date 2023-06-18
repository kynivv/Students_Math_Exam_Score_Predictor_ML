import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Data Importing
df = pd.read_csv('exams.csv')


# EDA
#print(df.isnull().sum())
#print(df.dtypes)


# Data Transformation
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

#print(df.dtypes)


# Training Test Split
from sklearn.model_selection import train_test_split

features = df.drop('math score', axis=1)
target = df['math score']
#print(features, target)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.23, random_state=22)
#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Model Training
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, StackingRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.svm import LinearSVR
from sklearn.metrics import explained_variance_score as evs

estimators = [
     ('lr', RidgeCV()),
     ('svr', LinearSVR(random_state=42))
 ]

models = [RandomForestRegressor(), LinearRegression(), RidgeCV(), LinearSVR(), DecisionTreeRegressor(), GradientBoostingRegressor(), BaggingRegressor(), StackingRegressor(estimators=estimators), VotingRegressor(estimators=estimators)]

for m in models:
    m.fit(X_train, Y_train)
    pred_train = m.predict(X_train)
    print(f'Train Accuracy of {m} is : {evs(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy of {m} is : {evs(Y_test, pred_test)}')
    print('|')
    print('|')
    print('|')


# Linear Regression shows the best results