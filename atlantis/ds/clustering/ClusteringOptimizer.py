from pandas import DataFrame
from joblib import Parallel, delayed
from math import sqrt
import matplotlib.pyplot as plt

from ...time.progress import ProgressBar, iterate
from ...time import get_elapsed, get_now
from ...exceptions import ClusteringError

from .KMeans import KMeans


class ClusteringOptimizer:
	def __init__(
			self, min_k, max_k, num_jobs=-1, keep_models=False, timeout=None,
			external_parallelization=True, num_external_jobs=None, num_internal_jobs=None, **kwargs
	):
		self._min_k = min_k
		self._max_k = max_k
		self._num_jobs = num_jobs
		self._models = None
		self._kwargs = kwargs
		self._keep_models = keep_models
		self._distortions = None
		self._inertias = None
		self._silhouette_scores = None
		self._timeout = timeout
		self._external_parallelization = external_parallelization
		self._num_external_jobs = num_external_jobs
		self._num_internal_jobs = num_internal_jobs

	def fit(self, X, echo=1, raise_timeout=True, y=None):
		if self._keep_models:
			self._models = {}

		self._distortions = {}
		self._inertias = {}
		self._silhouette_scores = {}

		def get_scores(k, num_jobs):
			kmeans = KMeans(
				n_clusters=k, n_jobs=num_jobs, timeout=self._timeout, raise_timeout=raise_timeout, **self._kwargs
			)
			kmeans.fit(X=X)
			if kmeans.timedout:
				return k, None, None, None

			if self._keep_models:
				return k, kmeans, kmeans.inertia, kmeans.distortion, kmeans.silhouette_score

			else:
				return k, None, kmeans.inertia, kmeans.distortion, kmeans.silhouette_score

		if self._external_parallelization and self._num_jobs > 1:

			num_ks = self._max_k - self._min_k + 1
			num_external_jobs = self._num_external_jobs or min(num_ks, self._num_jobs)
			remaining_jobs = self._num_jobs - num_external_jobs
			num_internal_jobs = self._num_internal_jobs or max(1, int(round(remaining_jobs / num_ks)))
			print(f'num_ks = {num_ks} num_external_jobs = {num_external_jobs} num_internal_jobs = {num_internal_jobs}')

			processor = Parallel(n_jobs=num_external_jobs, backend='threading', require='sharedmem')
			result = processor(
				delayed(get_scores)(k=k, num_jobs=num_internal_jobs)
				for k in iterate(iterable=range(self._max_k, self._min_k - 1, -1), echo_items=True)
			)

			for k, model, inertia, distortion, silhouette_score in result:
				if inertia is not None or distortion is not None or silhouette_score is not None:
					self._distortions[k] = distortion
					self._inertias[k] = inertia
					self._silhouette_scores[k] = silhouette_score
					if self._keep_models:
						self._models[k] = model
				else:
					raise ClusteringError(f'Both inertia and distortion are None for k: {k}')

		else:

			progress_bar = ProgressBar(total=self._max_k - self._min_k + 1, echo=echo)
			error = ''

			elapsed = 0
			for k in range(self._max_k, self._min_k - 1, -1):
				start_time = get_now()
				progress_bar.show(amount=self._max_k - k, text=f'j={self._num_jobs}, k={k}{error}, t={round(elapsed)}')
				k, model, inertia, distortion, silhouette_score = get_scores(k=k, num_jobs=self._num_jobs)
				if inertia is not None or distortion is not None or silhouette_score is not None:
					if self._keep_models:
						self._models[k] = model
					self._distortions[k] = distortion
					self._inertias[k] = inertia
					self._silhouette_scores[k] = silhouette_score
				else:
					raise ClusteringError(f'Both inertia and distortion are None for k: {k}')
				elapsed = max(elapsed, get_elapsed(start=start_time, unit='s'))

			progress_bar.show(amount=progress_bar.total)

	def get_scores(self, by='silhouette_score'):
		"""
		:rtype: dict
		"""
		if by.startswith('silhouette'):
			return self._silhouette_scores.copy()
		elif by.startswith('inertia'):
			return self._inertias.copy()
		elif by.startswith('distortion'):
			return self._distortions.copy()

	def plot(self, by='silhouette_score', size=(16, 8)):
		scores = self.get_scores(by=by)
		plt.figure(figsize=size)
		plt.plot(list(scores.keys()), list(scores.values()), 'bx-')
		plt.xlabel('k')
		plt.ylabel(by.replace('_', ' ').capitalize())
		plt.title('The Elbow Method showing the optimal k')
		plt.show()

	def get_optimal_number_of_clusters(self, by='silhouette'):
		"""
		:param str by: can be 'inertia' or 'distortion'
		:rtype: int
		"""
		x1, x2 = min(self._silhouette_scores.keys()), max(self._silhouette_scores.keys())

		if by.startswith('silhouette'):
			return max(self._silhouette_scores, key=self._silhouette_scores.get)

		if by.startswith('inertia'):
			y_dictionary = self._inertias

		elif by.startswith('distort'):
			y_dictionary = self._distortions

		else:
			y_dictionary = None
			ValueError(f'Method "{by}" is not supported!')

		# Elbow method
		y1, y2 = y_dictionary[x1], y_dictionary[x2]
		distances = []
		for x in range(x1, x2 + 1):
			y = y_dictionary[x]

			numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
			denominator = sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
			distances.append(numerator / denominator)

		return distances.index(max(distances)) + 2

	@property
	def optimal_number_of_clusters(self):
		"""
		:rtype: int
		"""

		scores = []
		for by in ['silhouette', 'inertia', 'distortion']:
			try:
				score = self.get_optimal_number_of_clusters(by=by)
				scores.append(score)
			except TypeError:
				continue

		return max(scores)

	@property
	def records(self):
		"""
		:rtype: list[dict]
		"""
		return [
			{'k': k, 'distortion': self._distortions[k], 'inertia': self._inertias[k]}
			for k in range(self._min_k, self._max_k + 1)
		]

	@property
	def data(self):
		"""
		:rtype: DataFrame
		"""
		return DataFrame.from_records(self.records)
