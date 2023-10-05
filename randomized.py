import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    dataset = pd.read_csv("./data/felicidad.csv")
    print(dataset)

    X = dataset.drop(["country","rank","score"], axis=1) #rank is tooo correlated to score
    y = dataset["score"]

    reg = RandomForestRegressor()

    parametros = {
        "n_estimators" : range(4,16), #from 4 to 15 trees
        "criterion" : ['absolute_error', 'squared_error'],
        "max_depth" : range(2,11) # how deep the tree is
    }

    rand_est = RandomizedSearchCV(reg, parametros, n_iter=10, cv=3, scoring = "neg_mean_absolute_error").fit(X,y)
    #n_iter means how many randomized parameters selection we are going to use

    print(rand_est.best_estimator_) #best parameters taking in consideration scoring
    print(rand_est.best_params_)
    print(rand_est.predict(X.loc[[0]])) #for predict, as a meta estimator he would use the best scored
    print(y[0])


