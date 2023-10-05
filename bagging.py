import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    
    dt_heart = pd.read_csv("./data/Heart.csv")
    print(dt_heart["target"].describe()) #statistic description

    X = dt_heart.drop(["target"], axis = 1) # .drop copys an entire
    # dataset to save inside another var, without a selected column. However,
    #  using the parameter inplace = "True", we can just remove the selected var
    # from the original dataset. 
    y = dt_heart["target"] #this creation of new datset is made, 
    #to ensure no modification of original dataset, because features 
    # and target is needed, in case we modify and drop original target
    #from original dataset, we can not save it inside y later.

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35)

    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    print("="*64)
    print(accuracy_score(knn_pred,y_test))


    bag_class = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    #baggin for classification, estimator for setting the base estimator for this method, n_estimator
    #defines the number of estimator we are using for baggin.
    bag_pred = bag_class.predict(X_test)
    print("="*64)
    print(accuracy_score(bag_pred,y_test))

    #From this method we can conclude a lot of stuff, fist of all. Base estimator was not the ideal
    #for this problem from the beggining, however using bagging we could assure a significately higger
    #performance for the classifier.

    #parameters for bagging can be tuned for performance boosting.




