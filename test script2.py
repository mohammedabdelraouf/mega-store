from sklearn.model_selection import train_test_split
import preprocessing as pre
import pickle as pick
import pandas as pd

# classification
classification_data = pd.read_csv("megastore-tas-test-classification.csv")
selected_features2 = pick.load(open('models/features2.pkl', 'rb'))

Y2 = classification_data.loc[:, 'ReturnCategory']
X2 = classification_data.drop('ReturnCategory', axis=1)

X2 = pre.pre_processing(X2)
X2 = X2[selected_features2.columns]
X_test_num, X_test_cat = pre.numerical_Categorical(X2)
statistics = pick.load(open('models/statistics2.pkl', 'rb'))
# handel Na values
skew_values = X_test_num.skew()
for column in X_test_num.columns[X_test_num.isnull().any()]:
    skewness = skew_values[column]
    if abs(skewness) < 0.5:  # Assuming a skewness threshold of 0.5
        X2[column].fillna(statistics.at[column, 'mean'], inplace=True)
    else:
        X2[column].fillna(statistics.at[column, 'median'], inplace=True)


for col in X_test_cat:
    encoder = pick.load(open('encoders/' + col + '2.sav', 'rb'))
    df = encoder.inverse_transform(pd.Series([statistics.at[col, 'mode']], name=col, index=[0]))
    X2[col].fillna(df.at[0, col])
    X2.loc[:, col] = encoder.transform(X2.loc[:, col])
encoder1 = pick.load(open('encoders/ReturnCategory.sav', 'rb'))
df = encoder1.inverse_transform(pd.Series([statistics.at['ReturnCategory', 'mode']], name='ReturnCategory', index=[0]))
Y2.fillna(df.at[0, 'ReturnCategory'])
Y2 = encoder1.transform(Y2)

# X_train, X_test, y_train, y_test = train_test_split(X2, Y2, test_size=1, shuffle=True, random_state=1)

Logistic_Regression = pick.load(open("models/LogisticRegression.sav", "rb"))
print("Logistic regression score :", Logistic_Regression.score(X2, Y2))

Bagging_Classifier = pick.load(open("models/BaggingClassifier.sav", "rb"))
print("Bagging classifier score  :", Bagging_Classifier.score(X2, Y2))

Random_ForestClassifier = pick.load(open("models/RandomForestClassifier.sav", "rb"))
print("Random forest classifier score  :", Random_ForestClassifier.score(X2, Y2))

SVM = pick.load(open("models/SVMClassifier.sav", "rb"))
print("SVM classifier score  :", SVM.score(X2, Y2))

decision_tree = pick.load(open("models/DecisionTreeClassifier.sav", "rb"))
print("Decision tree classifier score  :", decision_tree.score(X2, Y2))