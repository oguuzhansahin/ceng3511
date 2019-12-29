import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


class KNN:
    
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]  
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

if __name__ == "__main__":

    train = np.loadtxt("train.txt",delimiter=",",skiprows=1)
    X_train,y_train = train[...,:-1],train[...,-1]
    
    test = np.loadtxt("test.txt",delimiter=",",skiprows=1)
    test_labels = test[...,:-1]
    y_true = test[...,-1]
    
    clf = KNN(k=10)
    clf.fit(X_train,y_train)
    predictions = clf.predict(test_labels)
    print("custom KNN classification accuracy", accuracy(y_true, predictions))
    
    
    
    





    
    
    

