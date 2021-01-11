from sklearn.cluster import KMeans as _KMeans
from func_timeout import func_timeout, FunctionTimedOut
from pandas import DataFrame, concat

from .get_distortion import get_distortion


class TimedOutKMeans:
	pass


class KMeans:
	def __init__(self, num_clusters, num_jobs=-1, timeout=None, **kwargs):
		self._num_clusters = num_clusters
		self._num_jobs = num_jobs
		self._kmeans = _KMeans(n_clusters=num_clusters, n_jobs=num_jobs, **kwargs)
		self._distortion = None
		self._inertia = None
		self._timeout = timeout
		self._timedout = False

	def fit(self, X, y=None, raise_timeout=True):
		if self._timeout:
			try:
				func_timeout(timeout=self._timeout, func=self._kmeans.fit, kwargs={'X': X})
				self._distortion = get_distortion(X=X, kmeans_model=self._kmeans)
				self._inertia = self._kmeans.inertia_
			except FunctionTimedOut as e:
				self._timedout = True
				if raise_timeout:
					raise e
				else:
					self._kmeans = TimedOutKMeans()

		else:
			self._kmeans.fit(X=X)
			self._distortion = get_distortion(X=X, kmeans_model=self._kmeans)
			self._inertia = self._kmeans.inertia_

	@property
	def num_clusters(self):
		"""
		:rtype: int
		"""
		return self._num_clusters

	@property
	def timedout(self):
		return self._timedout

	@property
	def distortion(self):
		return self._distortion

	@property
	def inertia(self):
		return self._inertia

	def predict(self, X, prefix='cluster_'):
		"""
		:type X: DataFrame
		:type prefix: str
		:rtype: list[str]
		"""
		return [f'{prefix}{x + 1}' for x in self._kmeans.predict(X=X)]

	def transform(self, X, prefix='distance_'):
		"""
		:type X: DataFrame
		:type prefix: str
		:rtype: DataFrame
		"""
		result = DataFrame(self._kmeans.transform(X=X), columns=[f'{prefix}{x + 1}' for x in range(self._num_clusters)])
		if isinstance(X, DataFrame):
			result.index = X.index
		return result

	def transform_and_predict(self, X, append=False):
		"""
		:type X: DataFrame
		:type append: bool
		:rtype: DataFrame
		"""
		transformation = self.transform(X=X)
		transformation['prediction'] = self.predict(X=X)
		if append:
			X = X.copy()
			X = concat([X, transformation], axis=1)
			return X
		else:
			return transformation
