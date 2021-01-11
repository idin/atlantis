from scipy.spatial.distance import cdist
import numpy as np


def get_distortion(X, kmeans_model):
	return sum(np.min(cdist(X, kmeans_model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
