import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA


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

    kpca = KernelPCA(n_components=4, kernel="poly")
    #kernel parameter must be declared for kpca case
    #some examples for "kernel" parameter -> linear, poly,rbf

    kpca.fit(X_train) #must be adjusted

    dt_train = kpca.transform(X_train) 
    dt_test = kpca.transform(X_test)

    logistic = LogisticRegression(solver="lbfgs") 
    #if not sure about which solver should be use, when pointing over 
    #the LogisticRegression() statement, a param solver will be suggest.

    logistic.fit(dt_train, y_train) #fit the logistic model to the transformed train data.
    print("SCORE KPCA: ", logistic.score(dt_test, y_test)) # get the score based on transformed test data.
     
     #Remember to run the code after activate virtual environment for developing.