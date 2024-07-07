import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# CLASSIFICATION MODEL
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def plot_confusion_matrix(y,y_predict):
    '''
        plots the confusion matrix
    '''
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show() 

def regresion_model(X_train, Y_train, X_test, Y_test):
    '''
        Logistic Regression model
    '''
    cv_param ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
    lr=LogisticRegression()
    
    # Predict model
    lr.fit(X_train, Y_train)
    pred = lr.predict(X_test)
    acc = lr.score(X_test, Y_test)

    # Get confusin matrix
    logreg_cv = GridSearchCV(lr, cv_param)
    logreg_cv.fit(X_train, Y_train)
    yhat=logreg_cv.predict(X_test)

    #plot_confusion_matrix(Y_test,yhat)
    
    return acc.round(2), yhat

def svm_model(X_train, Y_train, X_test, Y_test):
    '''
        Support Vectom Machine model
    '''
    cv_param = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
    svm = SVC()
    # For svm predict
    svm.fit(X_train , Y_train)
    pred = svm.predict(X_test)
    acc = svm.score(X_test , Y_test)
    # For confiuson matrix
    svm_cv = GridSearchCV(svm , cv_param)
    svm_cv.fit(X_train, Y_train)
    yhat=svm_cv.predict(X_test)

    return acc.round(2), yhat

def knn_model(X_train, Y_train, X_test, Y_test):
    '''
        K-Neabour Nearest model
    '''
    cv_param = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}
    # KNN predict
    KNN = KNeighborsClassifier()
    KNN.fit(X_train , Y_train)
    acc = KNN.score(X_test , Y_test)
    # Confusion matrix
    knn_cv = GridSearchCV(KNN , cv_param)
    knn_cv.fit(X_train , Y_train)
    yhat = knn_cv.predict(X_test)

    return acc.round(2), yhat

def Dtree_model(X_train, Y_train, X_test, Y_test):
    cv_param = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

    # Dession Tree prediction
    tree = DecisionTreeClassifier()
    tree.fit(X_train, Y_train)
    tree.predict(X_test)
    acc = tree.score(X_test , Y_test)
    # Confusion Matrix
    tree_cv = GridSearchCV(tree , cv_param)
    tree_cv.fit(X_train, Y_train)
    yhat = tree_cv.predict(X_test)

    return acc.round(2), yhat

def edit_outcome_values(val):
    if val.startswith('True'):
       new_value = 'True'
    else:
       new_value = 'False'
    return new_value

def plot_piechart(model_df):
    fig = go.Figure(
    data=[go.Pie(
        labels=model_df['model'],
        values=model_df['accuracy'],
        marker=dict(colors=['Gold', 'MediumTurquoise', 'LightGreen']),
        textinfo='text',
        texttemplate='%{label}<br>Accuray: %{value}',
        )]
    )
    fig.show() 

def main():
    # Read Data 
    data = pd.read_csv("dataset_collected.csv")
    df = pd.DataFrame(data)
    # change outcome to be two value (True, False) to suitable for classification model.
    df['Outcome'] = df['Outcome'].apply(edit_outcome_values)

    # separete dataset into feature and target for classification model 
    #X = df[['FlightNumber','customers','BoosterVersion','PayloadMass','Orbit','LaunchSite','GridFins','Reused','Legs','Block','ReusedCount']]
    X = df[['FlightNumber','PayloadMass','Block','ReusedCount']]
    Y = df['Outcome'].to_numpy()

    # Train the model
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state= 2)

    # ------------ MODELING ----------------  
    # Logistic Regression
    reg_acc, reg_yhat = regresion_model(X_train, Y_train, X_test, Y_test)   
    plot_confusion_matrix(Y_test,reg_yhat)

    # K-Nearest Neigbour 
    knn_acc, knn_yhat = knn_model(X_train, Y_train, X_test, Y_test)
    plot_confusion_matrix(Y_test, knn_yhat)

    # Decision Tree model 
    Dtree_acc, Dtree_yhat = Dtree_model(X_train, Y_train, X_test, Y_test)
    plot_confusion_matrix(Y_test, Dtree_yhat)

    # Create dataframe include all model ( accuracy and model name)
    model_df = pd.DataFrame(columns=['model','accuracy'])
    model_df['model'] = ['Logistic_Regression','K_Neareast_Neagbour','Desicion_Tree']
    model_df['accuracy'] = [reg_acc, knn_acc, Dtree_acc]
    
    # Create Piechart to identify modeling result
    plot_piechart(model_df)

if __name__ == '__main__':
  main()