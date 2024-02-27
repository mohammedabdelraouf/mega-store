import pickle
import time
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
import warnings
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import preprocessing as pre
import category_encoders as ce
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.linear_model import LogisticRegression

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

data = pd.read_csv("megastore-classification-dataset.csv")
Y = data.loc[:, 'ReturnCategory']
X = data.drop('ReturnCategory', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=42)

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

X_train = pre.pre_processing(X_train)
X_train_num, X_train_cat = pre.numerical_Categorical(X_train)
# encoding

#
for col in X_train_cat:
    encoder = ce.OrdinalEncoder(handle_unknown='value', handle_missing='value')
    X_train.loc[:, col] = encoder.fit_transform(X_train.loc[:, col])
    X_train_cat.loc[:, col] = X_train.loc[:, col]
    pickle.dump(encoder, open('encoders/' + col + "2" + '.sav', 'wb'))

encoder1 = ce.OrdinalEncoder(handle_unknown='value', handle_missing='value')
y_train = encoder1.fit_transform(y_train)
pickle.dump(encoder1, open('encoders/ReturnCategory.sav', 'wb'))

statistics = pd.concat([X_train.mean(axis=0, skipna=False),
                        X_train.median(axis=0, skipna=False)], axis=1)
statistics = pd.concat([statistics, X_train.mode(axis=0, numeric_only=False).loc[0, :]], axis=1)
tar = pd.Series([y_train.mean()[0], y_train.median()[0],
                 y_train.mode().loc[0, 'ReturnCategory']], name='ReturnCategory',
                index=['mean', 'median', 'mode'])
statistics.columns = ['mean', 'median', 'mode']
statistics = statistics.append(tar)
pickle.dump(statistics, open("models/statistics2.pkl", "wb"))

# feature selection
fs_train = SelectKBest(score_func=f_regression, k=4)
features_num_score = pd.concat([pd.DataFrame(X_train_num.columns),
                                pd.DataFrame(fs_train.fit(X_train_num, y_train).scores_)], axis=1)
features_num_score.columns = ['numerical Features', 'F_Score']
selected_num_features = features_num_score.nlargest(4, columns='F_Score')

L_train = []
for col in X_train_num.columns:
    if col not in selected_num_features['numerical Features'].values:
        L_train.append(col)
X_train_num.drop(L_train, axis=1, inplace=True)

chi_score = chi2(X_train_cat, y_train)

chi_values = pd.Series(chi_score[0], index=X_train_cat.columns)
chi_values.sort_values(ascending=False, inplace=True)


def plot_graph(cc):
    cc.plot.bar()
    plt.show()


plot_graph(chi_values)
chi_values = pd.Series(chi_score[1], index=X_train_cat.columns)
chi_values.sort_values(ascending=False, inplace=True)
plot_graph(chi_values)

features_chi_score = pd.DataFrame(chi_score[0])
features = pd.DataFrame(X_train_cat.columns)
features_chi_score = pd.concat([features, features_chi_score], axis=1)
features_chi_score.columns = ['Categorical Features', 'chi_Score']
selected_chi_features = features_chi_score.nlargest(7, columns='chi_Score')

L_train_cat = []
for col in X_train_cat.columns:
    if col not in selected_chi_features['Categorical Features'].values:
        L_train_cat.append(col)
X_train_cat.drop(L_train_cat, axis=1, inplace=True)

X_train = pd.concat([X_train_num, X_train_cat], axis=1)

print("selection done")
pickle.dump(X_train.loc[0:1, :], open('models/features2.pkl', 'wb'))


# end feature selection

# train
# select best parameter for decision tree


def bestParameters(X, Y, classifier, parameter1, list1, parameter2, value):
    x_train2, x_val, y_train2, y_val = train_test_split(X, Y, test_size=0.3, random_state=2)
    train_accuracy_values = []
    val_accuracy_values = []
    for p in list1:
        if value == -1:
            model = classifier(**{parameter1: p})
        else:
            model = classifier(**{parameter1: p, parameter2: value})
        model.fit(x_train2, y_train2)
        y_pred_train = model.predict(x_train2)
        y_pred_val = model.predict(x_val)
        acc_train = accuracy_score(y_train2, y_pred_train)
        acc_val = accuracy_score(y_val, y_pred_val)
        train_accuracy_values.append(acc_train)
        val_accuracy_values.append(acc_val)
    plt.plot(list1, train_accuracy_values, label='acc train')
    plt.plot(list1, val_accuracy_values, label='acc val')
    plt.legend()
    plt.grid(axis='both')
    plt.xlabel(parameter1 + ' parameter')
    plt.ylabel('accuracy')
    plt.title('Effect of entered parameter on accuracy')
    plt.show()


max_dep = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14]
max_features = [4, 5, 6, 7, 8, 9, 10, 11]
c = [0.000000000000000001, 1, 10, 100, 150]
bestParameters(X_train, y_train, DecisionTreeClassifier, "max_depth", max_dep, "max_features", 8)
bestParameters(X_train, y_train, DecisionTreeClassifier, "max_features", max_features, "max_depth", 7)
# bestParameters(X_train, y_train, svm.SVC, "C", c, "kernel", -1)
start_time = time.time()
# t = time.time()
model = LogisticRegression().fit(X_train, y_train)
# print("1-- %s" % (time.time() - t))
# t = time.time()
model2 = BaggingClassifier(n_estimators=100, max_samples=6000).fit(X_train, y_train)
# print("2-- %s" % (time.time() - t))
# t = time.time()
model3 = RandomForestClassifier(n_estimators=100, max_samples=6000).fit(X_train, y_train)
# print("3-- %s" % (time.time() - t))
# t = time.time()
model4 = svm.SVC(C=150).fit(X_train, y_train)
# print("4-- %s" % (time.time() - t))
# t = time.time()
model5 = DecisionTreeClassifier(max_depth=7, max_features=11).fit(X_train, y_train)
# print("5-- %s" % (time.time() - t))

pickle.dump(model, open("models/LogisticRegression.sav", "wb"))
pickle.dump(model2, open("models/BaggingClassifier.sav", "wb"))
pickle.dump(model3, open("models/RandomForestClassifier.sav", "wb"))
pickle.dump(model4, open("models/SVMClassifier.sav", "wb"))
pickle.dump(model5, open("models/DecisionTreeClassifier.sav", "wb"))
# end train

print("training done time taken %s seconds " % (time.time() - start_time))

# testing
X_test = pre.pre_processing(X_test)
X_test_num, X_test_cat = pre.numerical_Categorical(X_test)
for col in X_test_cat:
    encoder = pickle.load(open('encoders/' + col + '2.sav', 'rb'))
    X_test_cat.loc[:, col] = encoder.transform(X_test_cat.loc[:, col])
encoder1 = pickle.load(open('encoders/ReturnCategory.sav', 'rb'))
y_test = encoder1.transform(y_test)

# feature selection as in training
X_test_cat.drop(L_train_cat, axis=1, inplace=True)
X_test_num.drop(L_train, axis=1, inplace=True)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)

# predictions
start_time = time.time()
# t = time.time()
print("Logistic regression train score :", model.score(X_train, y_train))
print("Logistic regression score :", model.score(X_test, y_test))
# print("1-- %s" % (time.time() - t))
# t = time.time()
print("\nBagging classifier train score :", model2.score(X_train, y_train))
print("Bagging classifier score  :", model2.score(X_test, y_test))
# print("2-- %s" % (time.time() - t))
# t = time.time()
print("\nRandom forest classifier train score :", model3.score(X_train, y_train))
print("Random forest classifier score  :", model3.score(X_test, y_test))
# print("3-- %s" % (time.time() - t))
# t = time.time()
print(f"\nSVM classifier train score:  {model4.score(X_train, y_train)}")
print(f"SVM classifier test score:  {model4.score(X_test, y_test)}")
# print("4-- %s" % (time.time() - t))
# t = time.time()
print(f"\nDecision tree classifier train score:  {model5.score(X_train, y_train)}")
print(f"Decision tree classifier test score:  {model5.score(X_test, y_test)}")
print("\ntesting done time taken %s seconds " % (time.time() - start_time))
# print("5-- %s" % (time.time() - t))
