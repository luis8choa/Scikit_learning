import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score, KFold #para importar dos funciones del mismo modo, solo ponemos ","
)

if __name__ == "__main__":

    dataset = pd.read_csv("./data/felicidad.csv")

    X = dataset.drop(["country","score"], axis =1)
    y = dataset["score"]

    model = DecisionTreeRegressor()
    score = cross_val_score(model,X,y,cv = 3,scoring="neg_mean_squared_error") 
    #recomended for quick testing, and you dont want to make additional configurations
    #scoring means the type of score que use to defined the better performed model
    #defaul cv (folds) are 5
    print(np.abs(np.mean(score))) #numpy method for mean of the 3 folds, make it easier to see

    #for seeing how does cv works internally
    kf = KFold(n_splits=3, shuffle=True, random_state=42) 
    #number of folds, random state fixed for replicability
    for train, test in kf.split(dataset):
        print(train)
        print(test)



