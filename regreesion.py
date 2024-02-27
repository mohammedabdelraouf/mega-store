import pickle
import time
import pandas as pd
import seaborn as sns
import warnings
import preprocessing as pre
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

data = pd.read_csv("megastore-regression-dataset.csv")
Y = data.loc[:, 'Profit']
X = data.drop('Profit', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=42)

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

X_train = pre.pre_processing(X_train)
X_train_num, X_train_cat = pre.numerical_Categorical(X_train)
# encoding
for col in X_train_cat:
    encoder = ce.OrdinalEncoder(handle_unknown='value', handle_missing='value')
    X_train.loc[:, col] = encoder.fit_transform(X_train.loc[:, col])
    X_train_cat.loc[:, col] = X_train.loc[:, col]
    pickle.dump(encoder, open('encoders/'+col+'.sav', 'wb'))
statistics = pd.concat([X_train.mean(axis=0, skipna=False),
                        X_train.median(axis=0, skipna=False)], axis=1)
statistics = pd.concat([statistics, X_train.mode(axis=0, numeric_only=False).loc[0, :]], axis=1)
tar = pd.Series([y_train.mean(), y_train.median(), y_train.mode()[0]], name='Profit', index=['mean', 'median', 'mode'])
statistics.columns = ['mean', 'median', 'mode']
statistics = statistics.append(tar)
pickle.dump(statistics, open("models/statistics1.pkl", "wb"))
# feature selection
fs_train = SelectKBest(score_func=f_regression, k=6)
feature_score = pd.concat([pd.DataFrame(X_train_cat.columns),
                           pd.DataFrame(fs_train.fit(X_train_cat, y_train).scores_)],
                          axis=1)
feature_score.columns = ['Categorical Features', 'F_Score']
feature_score = feature_score.nlargest(6, columns='F_Score')

L_train = []
for col in X_train_cat.columns:
    if col not in feature_score['Categorical Features'].values:
        L_train.append(col)


X_train_cat.drop(L_train, axis=1, inplace=True)

X_train_num.loc[:, 'Profit'] = y_train
# visualization of the correlation between features and target
p1 = plt
p1.subplots(figsize=(12, 8))
corr = X_train_num.corr()
sns.heatmap(corr, annot=True)
p1.show()

top_features = corr.index[abs(corr.loc[:, 'Profit']) > 0.2]
X_train_num = X_train_num[top_features].drop(columns=['Profit'])
X_train = pd.concat([X_train_num, X_train_cat], axis=1)
pickle.dump(X_train.loc[0:1, :], open('models/features.pkl', 'wb'))
# end feature selection

# transforms the existing features to higher degree features.
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
pickle.dump(poly_features, open("models/Polynomial.sav", "wb"))

# print(statistics.at['Sales', 'mean'])

# fit the transformed features to Linear Regression

# rid = Ridge()
# rid.fit(X_train, y_train)
# pickle.dump(rid, open("models/ridge_model.sav", "wb"))
#
# lass = Lasso()
# lass.fit(X_train, y_train)
# pickle.dump(lass, open("models/Lasso_model.sav", "wb"))
start_time = time.time()
t = time.time()
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
# print("1-- %s" % (time.time() - t))
# fit on other regression models to find the best model fit in our task
# t = time.time()
elasticNet = ElasticNet().fit(X_train, y_train)
# print("2-- %s" % (time.time() - t))
# t = time.time()
model = LinearRegression().fit(X_train, y_train)
# print("3-- %s" % (time.time() - t))
# t = time.time()
regr = RandomForestRegressor(max_depth=12, random_state=1).fit(X_train, y_train)
# print("4-- %s" % (time.time() - t))
# t = time.time()
pickle.dump(regr, open("models/random_forest.sav", "wb"))
pickle.dump(model, open("models/multivariable.sav", "wb"))
pickle.dump(elasticNet, open("models/elasticNet_model.sav", "wb"))
pickle.dump(poly_model, open("models/poly_model.sav", "wb"))


print("training done time taken %s seconds " % (time.time() - start_time))

# testing
X_test = pre.pre_processing(X_test)
X_test_num, X_test_cat = pre.numerical_Categorical(X_test)

for col in X_test_cat:
    encoder = pickle.load(open('encoders/' + col + '.sav', 'rb'))
    X_test_cat.loc[:, col] = encoder.transform(X_test_cat.loc[:, col])


X_test_cat.drop(L_train, axis=1, inplace=True)

X_test_num = X_test_num[top_features[:-1]]
X_test = pd.concat([X_test_num, X_test_cat], axis=1)
# start_time = time.time()
# t = time.time()
print("mean square error of elasticNet train data set:", mean_squared_error(y_train, elasticNet.predict(X_train)))
print("mean square error of elasticNet test data set:", mean_squared_error(y_test, elasticNet.predict(X_test)))
print("elasticNet score : ", elasticNet.score(X_test, y_test), "\n")
# print("1-- %s" % (time.time() - t))
# t = time.time()

print("mean square error of Linear-multivariable train data set:", mean_squared_error(y_train, model.predict(X_train)))
print("mean square error of Linear-multivariable test data set:", mean_squared_error(y_test, model.predict(X_test)))
print("Linear-multivariable model score  : ", model.score(X_test, y_test), "\n")
# print("2-- %s" % (time.time() - t))
# t = time.time()

def plot_plt(y, pred, label1, label2, model_name):
    plt.scatter(y, pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.title(model_name+' Performance')
    plt.show()


regr_pred = regr.predict(X_test)

print("mean sqr error of random-forest-regressor train data set:", mean_squared_error(y_train, regr.predict(X_train)))
print("mean square error of random forest regressor test data set:", mean_squared_error(y_test, regr_pred))
print("random forest regressor model score  : ", regr.score(X_test, y_test), "\n")
# print("3- %s" % (time.time() - t))
# t = time.time()
poly_test = poly_features.fit_transform(X_test)
prediction = poly_model.predict(poly_test)

print("mean square error of Polynomial train data set:", mean_squared_error(y_train, poly_model.predict(X_train_poly)))
print("mean square error of Polynomial test data set:", mean_squared_error(y_test, prediction))
print("Polynomial model Score : ", poly_model.score(poly_test, y_test), "\n")
# print("4-- %s" % (time.time() - t))
# t = time.time()

print("testing done time taken %s seconds " % (time.time() - start_time))

plot_plt(y_test, prediction, 'Actual Values', 'Predicted Values', 'Polynomial Regression')

plot_plt(y_test, regr_pred, 'Actual Values', 'Predicted Values', 'random forest')
