import pandas as pd

from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)

from sklearn.svm import SVR #svm - support vectorial machine , SVR - Support vector regressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__": #is this main script
    dataset = pd.read_csv("./data/felicidad_corrupt.csv")
    print(dataset.head(5))

    X = dataset.drop(["country", "score"],axis = 1) # axis = 1 - means columns
    y = dataset[["score"]]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
 
    estimadores = { #dictionary. it is composed of key: value
        "SVR" : SVR(gamma= "auto", C=1.0, epsilon=0.1),
        "RANSAC": RANSACRegressor(),
        "Huber" : HuberRegressor(epsilon=1.35)
    } 

    for name, estimador in estimadores.items(): #dict .items returns key value
        estimador.fit(X_train,y_train)
        predictions = estimador.predict(X_test)

        print("_"*64)

        print(name)
        print("MSE: " , mean_squared_error(y_test,predictions))

