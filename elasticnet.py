import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet

if __name__ == "__main__":
    dataset = pd.read_csv("./data/felicidad.csv")
    print(dataset.describe()) #statistic description for every column.

    X = dataset[["gdp", "family", "lifexp", "freedom", "corruption", "generosity", "dystopia"]] #sleceted features for prediction
    y = dataset[["score"]]#target

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
    regr = ElasticNet(random_state=0)
    regr.fit(X_train, y_train)

    y_predict = regr.predict(X_test)
    loss = mean_squared_error(y_test, y_predict)
    print("Elastic Loss: ", loss)

    print("="*32)

    print("Coef ELASTIC")
    print(regr.coef_)



