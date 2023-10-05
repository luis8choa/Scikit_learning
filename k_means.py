import pandas as pd

from sklearn.cluster import MiniBatchKMeans 
#this is a k_means variation which is helpful
# for some ocassion when we dont have a higher 
# computational power. Low processing and Rams.
# works similar consuming less resources 

if __name__ == "__main__":

    dataset = pd.read_csv("./data/candy.csv")
    print(dataset.head(10))

    X = dataset.drop("competitorname", axis = 1)

    kmeans = MiniBatchKMeans(n_clusters=4,batch_size=8).fit(X)
    print("Total de centros: ", len(kmeans.cluster_centers_)) 
    
    #a correct example for this implementation
    #would be that a store wants to agruppate 4
    #differents groups of candys that are 
    #likely to be similar in between the groups.

    #batch_size - indicates the number of data we
    #  are using to train our algorithm in every
    #  iteration

    print("Total de centros: ", len(kmeans.cluster_centers_)) 
    #cluster_center must represent the amount of groups we have.
    print("="*64)
    print(kmeans.predict(X)) #prediction returns
    # a numeric int array indicating the group for every candy

    dataset["Group"] = kmeans.predict(X)
    #this declaration sets a new column in the original dataset
    #for the cluster we defined using kmeans, in this case we have
    # to ensure that the number of rows for the dataset and
    # lenght of clustering array match.

    print(dataset)

