from sklearn import preprocessing, neighbors, svm, tree, linear_model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import warnings
import joblib
from Sound_Class import Sound_Class
warnings.filterwarnings('ignore')

Sound_Object = Sound_Class("silenceRemoved", sample_rate=9000)

# region Read data and split it

# load data
load_data = Sound_Object.load_data()

# split to X & Y
X, Y = Sound_Object.split_to_x_y(load_data)

# split to train and test -> 80: 20

X_train, X_test, y_train, y_test = Sound_Object.train_test_split_func(X, Y, 0.2, 51)

# endregion

# region Get MFCC Feature

X_train_mfcc = Sound_Object.get_feature(X_train)
X_test_mfcc = Sound_Object.get_feature(X_test)

# endregion


# region Models Implemented

# KNN
knn = neighbors.KNeighborsClassifier(n_neighbors=5)

# Support Vector Machine
svm = svm.SVC()

# Random Forest Classifier
rfc = RandomForestClassifier(random_state=50)

# Decision Tree
dt = tree.DecisionTreeClassifier(random_state=42, max_depth=9)



# AdaBoost Classifier
boost = AdaBoostClassifier(n_estimators=5)

# Logistic Regression
logistic = linear_model.LogisticRegression()

# endregion

# region Accuracy

print('MFCC:-')
# Train Accuracy
print('Train Accuracy: ')
Sound_Object.classifier('svm_classifier', svm, X_train_mfcc, y_train, X_train_mfcc, y_train)
Sound_Object.classifier('Decision Tree', dt, X_train_mfcc, y_train, X_train_mfcc, y_train)
Sound_Object.classifier('AdaBoostClassifier', boost, X_train_mfcc, y_train, X_train_mfcc, y_train)
Sound_Object.classifier('KNeighborsClassifier', knn, X_train_mfcc, y_train, X_train_mfcc, y_train)
Sound_Object.classifier('RandomForestClassifier', rfc, X_train_mfcc, y_train, X_train_mfcc, y_train)
Sound_Object.classifier('LogisticRegression', logistic, X_train_mfcc, y_train, X_train_mfcc, y_train)

print('-' * 50)

# Test Accuracy
print('Test Accuracy: ')
Sound_Object.classifier('svm_classifier', svm, X_train_mfcc, y_train, X_test_mfcc, y_test)
Sound_Object.classifier('Decision Tree', dt, X_train_mfcc, y_train, X_test_mfcc, y_test)
Sound_Object.classifier('AdaBoostClassifier', boost, X_train_mfcc, y_train, X_test_mfcc, y_test)
Sound_Object.classifier('KNeighborsClassifier', knn, X_train_mfcc, y_train, X_test_mfcc, y_test)
Sound_Object.classifier('RandomForestClassifier', rfc, X_train_mfcc, y_train, X_test_mfcc, y_test)
Sound_Object.classifier('LogisticRegression', logistic, X_train_mfcc, y_train, X_test_mfcc, y_test)

# endregion

# region save pipeline normalization and classifier models

pipeline = Pipeline([
    ('RandomForestClassifier', rfc)  # Classifier step
])
joblib.dump(pipeline, 'pipeline_with_normalization.joblib')

# endregion

# Show the Confusion Matrix of Random Forest Classifier as it is the best classifier
Sound_Object.ConfusionMatrix(rfc, X_test_mfcc, y_test)
