from numpy.core.fromnumeric import sort
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# read diabetes data
data = pd.read_csv('diabetes.csv')
data['Output'] = data['Glucose'].apply(
    lambda x: 2 if x > 125 else 1 if x > 99 and x <= 125 else 0)

# Remove the part where insulin !=0 for Col= 'Glucose','Output'
data1 = data.query('Insulin==0 & Glucose !=0')[['Glucose', 'Output']]

# Remove the part where insulin = 0 for Col= 'Glucose','Insulin','Output'
data2 = data.query("Insulin !=0 & Glucose !=0")[
    ['Glucose', 'Insulin', 'Output']]

train, test = train_test_split(data2, test_size=0.2, random_state=123)

X_train = train[[x for x in train.columns if x not in ["Insulin"]]]
y_train = train["Insulin"]

X_test = test[[x for x in test.columns if x not in ["Insulin"]]]
y_test = test["Insulin"]


model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

print("Train acc: ", model.score(X_train, y_train))
print("Test acc: ", model.score(X_test, y_test))

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, prediction))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, prediction))


# predict Insulin
prediction = model.predict(data1)
prediction = prediction.astype(int)

out = pd.DataFrame(prediction, columns=['Insulin'])

# predicted insulin values

dat = pd.concat([data1.reset_index(drop=True),
                 out.reset_index(drop=True)], axis=1)

new_data = pd.concat([dat, data2], sort=False)


# where Output == 2 change to 2 if Insulin < 30 elif change to 3 if insulin > 0
def myfunc(x, y):
    if x <= 30 and y == 2:
        return y
    elif x > 30 and y == 2:
        return y + 1
    else:
        return y


new_data['Output'] = new_data.apply(
    lambda x: myfunc(x.Insulin, x.Output), axis=1)

train, test = train_test_split(new_data, test_size=0.2, random_state=123)

X_train = train[[x for x in train.columns if x not in ["Output"]]]
y_train = train["Output"]

X_test = test[[x for x in test.columns if x not in ["Output"]]]
y_test = test["Output"]

model = KNeighborsClassifier()

model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)