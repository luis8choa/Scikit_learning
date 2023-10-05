import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
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

    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train,y_train) #n_estimators set the number of decision trees (smalls) 
    #we are going to use
    boost_pred = boost.predict(X_test)
    print("="*64)
    print(accuracy_score(boost_pred,y_test))

