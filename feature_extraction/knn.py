import numpy as np
from tqdm.auto import tqdm

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def knn_predict(X_train, y_train, X_test, k):
    predictions = []

    for test_sample in tqdm(y_train):
        distances = [euclidean_distance(test_sample, train_sample) for train_sample in X_train]
        nearest_neighbors_indices = np.argsort(distances)[:k]
        nearest_neighbors_labels = X_test[nearest_neighbors_indices]

        prediction = np.bincount(nearest_neighbors_labels).argmax()
        predictions.append(prediction)

    return np.array(predictions)


