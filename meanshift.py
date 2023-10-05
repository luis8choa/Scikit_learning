import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == "__main__":
    
    dataset = pd.read_csv("./data/candy.csv")
    print(dataset.head(5))

    X = dataset.drop("competitorname", axis=1)

    meanshift = MeanShift().fit(X) # meanshift  configuration 
    # has an important parameter call bandwidth. By default
    # if not specificied, automatic calculations are made 
    # to define the best bandwidth

    print("="*64)

    print(max(meanshift.labels_)) #defined labels for clustering. easiest way to determine 
    # how many classes are is using max() fuction

    print("="*64)

    print(meanshift.cluster_centers_) #centers for the distribution for the data
    #around them, returns an array for the lenght of the feature number of our data

    dataset["meanshift"] = meanshift.labels_

    print("="*64)

    print(dataset)
    
