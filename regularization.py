import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv("./data/felicidad.csv")
    print(dataset.describe()) #statistic description for every column.

    X = dataset[["gdp", "family", "lifexp", "freedom", "corruption", "generosity", "dystopia"]] #sleceted features for prediction
    y = dataset[["score"]]#target

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

    modelLinear = LinearRegression().fit(X_train,y_train) 
    y_predict_linear = modelLinear.predict(X_test)

    modelLasso = Lasso(alpha=0.02).fit(X_train,y_train) #bigger the alpha (lambda) bigger penalization.
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=1).fit(X_train,y_train) #fitting process
    y_predict_ridge = modelRidge.predict(X_test)

    linear_loss = mean_squared_error(y_test, y_predict_linear) #mean_square_error is a loss type calculation for expressing how far prediction is from the real value.
    print("Linear Loss: ", linear_loss)

    lasso_loss =  mean_squared_error(y_test,y_predict_lasso)
    print("Lasso Loss: ", lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge Loss: ", ridge_loss)

    print("="*32)

    print("Coef LINEAR")
    print(modelLinear.coef_)

    print("Coef LASSO")
    print(modelLasso.coef_) #.coef return the weights coefficients for the model.

    print("Coef RIDGE")
    print(modelRidge.coef_) #.coef return the weights coefficients for the model.







