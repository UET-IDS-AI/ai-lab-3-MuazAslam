"""
Linear & Logistic Regression Lab

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 everywhere required.
"""

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():
    x,y= load_diabetes(return_X_y=True)
    
    X_train , X_test ,Y_train , Y_test = train_test_split(x,y,test_size=0.20 , random_state=42)

    scaler=StandardScaler()

    X_train_optimize = scaler.fit_transform(X_train)

    X_test_optimize=scaler.transform(X_test)
    linear_regression=LinearRegression()

    linear_regression.fit(X_train_optimize , Y_train)

    y_pred = linear_regression.predict(X_test_optimize)
    y_train_pred = linear_regression.predict(X_train_optimize)

    test_mse  = mean_squared_error(Y_test,y_pred)

    train_mse = mean_squared_error(y_train_pred , Y_train)

    train_r2 = r2_score(y_train_pred , Y_train)

    test_r2=r2_score(Y_test , y_pred)

    coeffecients = linear_regression.coef_

    absolute_coff = abs(coeffecients)

    top_3_features = np.argsort(absolute_coff)[-3:][::-1]

    return train_mse , test_mse , train_r2 ,test_r2 ,top_3_features




    
    """
    
    STEP 2: Split into train and test (80-20).
            Use random_state=42.
    STEP 3: Standardize features using StandardScaler.
            IMPORTANT:
            - Fit scaler only on X_train
            - Transform both X_train and X_test
    STEP 4: Train LinearRegression model.
    STEP 5: Compute:
            - train_mse
            - test_mse
            - train_r2
            - test_r2
    STEP 6: Identify indices of top 3 features
            with largest absolute coefficients.

    RETURN:
        train_mse,
        test_mse,
        train_r2,
        test_r2,
        top_3_feature_indices (list length 3)
    """

    raise NotImplementedError


# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():
    X ,Y =load_diabetes(return_X_y=True)

    scaler=StandardScaler()

    linear_regression=LinearRegression()

    X_train ,X_test ,Y_train ,Y_test =train_test_split(X,Y,random_state=42)

    X_train_scaled=scaler.fit_transform(X_train)

    scores = cross_val_score(estimator=linear_regression,X=X_train_scaled ,y=Y_train , cv=5 , scoring="r2")

    r2_mean = np.mean(scores)

    r2_standard = np.std(scores)


    return r2_mean , r2_standard
    
    """
    STEP 1: Load diabetes dataset.
    STEP 2: Standardize entire dataset (after splitting is NOT needed for CV,
            but use pipeline logic manually).
    STEP 3: Perform 5-fold cross-validation
            using LinearRegression.
            Use scoring='r2'.

    STEP 4: Compute:
            - mean_r2
            - std_r2

    RETURN:
        mean_r2,
        std_r2
    """

    raise NotImplementedError


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():
    X,Y=load_breast_cancer(return_X_y=True)

    X_train ,x_test ,y_train ,y_test = train_test_split(X,Y , test_size=0.20 ,random_state=42)

    scaler=StandardScaler()


    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled=scaler.transform(x_test)

    LogisticRegressor =LogisticRegression(max_iter=5000)

    LogisticRegressor.fit(X_train_scaled , y_train)

    Y_train_pred = LogisticRegressor.predict(X_train_scaled)

    Y_test_pred =LogisticRegressor.predict(X_test_scaled)

    train_accuracy =accuracy_score(y_train ,Y_train_pred)

    test_accuracy =accuracy_score(y_test ,Y_test_pred)

    precision=precision_score(y_test , Y_test_pred)

    recall=recall_score(y_test , Y_test_pred)

    f1=f1_score(y_test , Y_test_pred)

    confusion__matrix=confusion_matrix(y_test , Y_test_pred)

    #False Negative is dangerous madically as it means that a test is Negative but it is False means an test goes undetected this is dangerous as if a person disease is not detetced timely then it will be dangerous and can be life threatening
    
    
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Split into train-test (80-20).
            Use random_state=42.
    STEP 3: Standardize features.
    STEP 4: Train LogisticRegression(max_iter=5000).
    STEP 5: Compute:
            - train_accuracy
            - test_accuracy
            - precision
            - recall
            - f1
            - confusion matrix (optional to compute but not return)

    In comments:
        Explain what a False Negative represents medically.

    RETURN:
        train_accuracy,
        test_accuracy,
        precision,
        recall,
        f1
    """
    return train_accuracy , test_accuracy , precision , recall , f1

    raise NotImplementedError

    


# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():

    X,Y= load_breast_cancer(return_X_y=True)

    X_train , X_test ,Y_train ,Y_test = train_test_split(X,Y,test_size=0.20,random_state=42)

    scaler=StandardScaler()

    X_train_scaled=scaler.fit_transform(X_train)

    X_test_scaled=scaler.transform(X_test)

    c_values = [0.01 , 0.1 , 1 ,10 , 100]

    logistic_regressor=LogisticRegression()

    results_dictionary={}

    for i in c_values : 
        logistic_regressor=LogisticRegression(max_iter=5000 , C=i)
        logistic_regressor.fit(X_train_scaled , Y_train)
        y_train_pred=logistic_regressor.predict(X_train_scaled)
        y_test_pred=logistic_regressor.predict(X_test_scaled)
        train_accuracy = accuracy_score(Y_train , y_train_pred)
        test_accuracy = accuracy_score(Y_test , y_test_pred)

        results_dictionary[i] = (train_accuracy , test_accuracy)
    
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Split into train-test (80-20).
    STEP 3: Standardize features.
    STEP 4: For C in [0.01, 0.1, 1, 10, 100]:
            - Train LogisticRegression(max_iter=5000, C=value)
            - Compute train accuracy
            - Compute test accuracy

    STEP 5: Store results in dictionary:
            {
                C_value: (train_accuracy, test_accuracy)
            }

    In comments:
        - What happens when C is very small?
        - What happens when C is very large?
        - Which case causes overfitting?

    RETURN:
        results_dictionary
    """

    # Comments:
    # Small C → stronger regularization → underfitting → lower accuracy
    # Large C → weaker regularization → overfitting → train accuracy high, test may drop
    # Overfitting occurs at large C values

    return results_dictionary

    raise NotImplementedError


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():

    X,Y=load_breast_cancer(return_X_y=True)
    
    logistic_regressor=LogisticRegression(C=1 , max_iter=5000)

    scaler=StandardScaler()

    X_train ,X_test ,Y_train ,Y_test = train_test_split(X,Y ,random_state=42)

    X_train_scaled=scaler.fit_transform(X_train)

    scores = cross_val_score(estimator=logistic_regressor, cv=5, X=X_train_scaled, y=Y_train, scoring="accuracy")

    mean_accuracy = np.mean(scores)

    std_accuracy =np.std(scores)
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Standardize entire dataset.
    STEP 3: Perform 5-fold cross-validation
            using LogisticRegression(C=1, max_iter=5000).
            Use scoring='accuracy'.

    STEP 4: Compute:
            - mean_accuracy
            - std_accuracy

    In comments:
        Explain why cross-validation is especially
        important in medical diagnosis problems.

    RETURN:
        mean_accuracy,
        std_accuracy

    """

    # Cross-validation is important in medical diagnosis because it ensures
    # the model generalizes well to unseen patients. This reduces the risk
   # of misdiagnosis and provides a more reliable estimate of accuracy.

    return mean_accuracy , std_accuracy

    raise NotImplementedError
