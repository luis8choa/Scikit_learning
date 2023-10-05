import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dt_heart = pd.read_csv("./data/Heart.csv")

    print(dt_heart.head(5)) #print head of the dataframe to ensure data was read.

    dt_features = dt_heart.drop(["target"], axis = 1) #this operation willbe carried over columns axis (1)
    dt_target = dt_heart["target"] #extract this feature
    
    dt_features = StandardScaler().fit_transform(dt_features) #fir_transform() -> load the data, adjust the model for 0-1 transformation and apply the transformation over the dataframe
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42) #train_test_split() -> defines partition for train and test,  must be applied over features and target, test_size of 0.3 (30%) subsequently indicates a train_size of (0.7). a fixed Random_state provides reproducibilty over partition over and over.   

    print(X_train.shape) #parametro .shape es propio de los datrafame de pandas
    print(y_train.shape) 

    #n_componentes por default = min(n_muestras, n_features), esto no reduce el numero de features en absoluto
    pca = PCA(n_components=3) #configuracion PCA
    pca.fit(X_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10) #un parametros adicional con relacion a PCA, un batch_size. 
    ipca.fit(X_train)

    plt.plot(range(len(pca.explained_variance_)),pca.explained_variance_ratio_) 
    #mostramos tomando el rango desde la longitud de los componentes que genero PCA al llamar a su metodo.
    #y luego la importancia de estos componentes.
    plt.show()

    logistic = LogisticRegression(solver="lbfgs") 
    #si se intenta correr la funcion LogisticRegression sin un solver, 
    #la funcion nos devuelve un solver adecuado para resolverlo.

    dt_train = pca.transform(X_train) # se aplica la transformada
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train) #fit recorre un modifica los datos, se pueden mandar dos dataframes a la vez 
    print("SCORE PCA: ", logistic.score(dt_test, y_test)) 
    #funcion interna de score tanto para datos prueba transformados por PCA,
    #como para datos de prueba sin transformacion.

    #ahora con ipca
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train,y_train)
    print("SCORE IPCA: ", logistic.score(dt_test, y_test)) 