from func_timeout import func_timeout, FunctionTimedOut
from numpy import array as np_array
from multiprocessing.pool import ThreadPool

from pandas import DataFrame as PandasDF, Series as PandasSeries, concat
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

from pyspark.sql import functions as f
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql import DataFrame as SparkDF, Column as SparkColumn
from pyspark.sql.window import Window
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors

from scipy.spatial.distance import cdist
import numpy as np


def get_distortion(X, kmeans_model):
	return sum(np.min(cdist(X, kmeans_model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]


class TimedOutKMeans:
	pass


class KMeans:
	def __init__(
			self, n_clusters, n_jobs=-1, timeout=None, evaluator=None, raise_timeout=True, **kwargs
	):
		"""
		:type n_clusters: int
		:type n_jobs: int
		:type env: str
		:type features: list[str]
		:type X: SparkDF or PandasDF
		:type raise_timeout: bool
		:type evaluator: ClusteringEvaluator
		"""
		self._n_clusters = n_clusters
		self._n_jobs = n_jobs
		self._kwargs = kwargs

		self._model_dict =None
		self._env = None
		self._features = None

		self._evaluator = evaluator
		self._raise_timeout = raise_timeout

		self._timeout = timeout
		self._timed_out = False

	############## PYTHON #################

	def _fit_python(self, X):
		"""
		:type X: PandasDF
		"""
		self._features = list(X.columns)
		model = SklearnKMeans(
			n_clusters=self.n_clusters, n_jobs=self._n_jobs, **self._kwargs
		)
		if self._timeout:
			try:
				func_timeout(timeout=self._timeout, func=model.fit, kwargs={'X': X})
				distortion = get_distortion(X=X, kmeans_model=model)
				inertia = model.inertia_

			except FunctionTimedOut as e:
				self._timed_out = True
				if self._raise_timeout:
					raise e
				else:
					model = TimedOutKMeans()
				distortion = None
				inertia = None
		else:
			model.fit(X=X)
			distortion = get_distortion(X=X, kmeans_model=model)
			inertia = model.inertia_
			predictions = model.predict(X=X)
			score = silhouette_score(X, predictions, metric='euclidean')

		self._model_dict = {
			'kmeans': model,
			'distortion': distortion,
			'inertia': inertia,
			'silhouette_score': score
		}
		return self

	@property
	def _python_model(self):
		"""
		:rtype: SklearnKMeans
		"""
		return self._model_dict['kmeans']

	@property
	def _python_cluster_centers(self):
		"""
		:rtype: list[np_array]
		"""
		return list(self._python_model.cluster_centers_)

	def _transform_python(self, X, prefix='distance_', keep_columns=False):
		"""
		:type X: PandasDF
		:type prefix: str
		:rtype: PandasDF
		"""
		result = PandasDF(
			self._python_model.transform(X=X[self._features]),
			columns=[f'{prefix}{x + 1}' for x in range(self._n_clusters)]
		)
		if isinstance(X, PandasDF):
			result.index = X.index

		if not keep_columns:
			return result

		elif isinstance(keep_columns, (list, str)):
			if isinstance(keep_columns, str):
				keep_columns = [keep_columns]
		else:
			keep_columns = list(X.columns)

		return concat([X[keep_columns], result], axis=1)

	def _predict_python(self, X, prefix='cluster_', keep_columns=False, return_dataframe=False):
		"""
		:type X: PandasDF
		:type prefix: str
		:type keep_columns: bool
		:type return_dataframe: bool
		:rtype: PandasSeries or PandasDF
		"""
		ndarray = self._python_model.predict(X=X[self._features])
		series = PandasSeries(ndarray, name='prediction')
		predictions = series.apply(lambda x: f'{prefix}{x + 1}')
		if keep_columns:
			if isinstance(keep_columns, str):
				keep_columns = [keep_columns]

			result = X.copy()
			if isinstance(keep_columns, list):
				result = result[keep_columns]

			result['prediction'] = predictions
			return result

		elif return_dataframe:
			return predictions.to_frame(name='prediction')

		else:
			return predictions

	############## SPARK #################

	def _fit_spark(self, X):
		"""
		:type X: SparkDF
		"""
		self._features = X.columns
		assembler = VectorAssembler(inputCols=self._features, outputCol="features")
		assembled = assembler.transform(X)
		kmeans = SparkKMeans(k=self.n_clusters).fit(assembled.select('features'))
		if self._evaluator is not None:
			clustered = kmeans.transform(assembled)
			silhouette_score = self._evaluator.evaluate(clustered)
			inertia = kmeans.summary.trainingCost

		else:
			silhouette_score = None
			inertia = None

		self._model_dict = {
			'assembler': assembler,
			'kmeans': kmeans,
			'silhouette_score': silhouette_score,
			'inertia': inertia
		}
		return self

	@property
	def _spark_model(self):
		"""
		:rtype: SparkKMeans
		"""
		return self._model_dict['kmeans']

	@property
	def _spark_assembler(self):
		"""
		:rtype: VectorAssembler
		"""
		return self._model_dict['assembler']

	@property
	def _spark_cluster_centers(self):
		"""
		:rtype: list[np_array]
		"""
		return self._spark_model.clusterCenters()

	def _transform_spark(self, X, prefix='distance_', keep_columns=False):
		"""
		:type X: SparkDF
		:type prefix: str
		:rtype: SparkDF
		"""
		assembled = self._spark_assembler.transform(X)
		centers = self.cluster_centers
		@f.udf(DoubleType())
		def _get_distance(features, cluster):
			"""
			:type features: np_array
			:rtype: SparkColumn
			"""
			center = centers[cluster]
			return float(Vectors.squared_distance(features, center) ** 0.5)

		if isinstance(keep_columns, (list, str)):
			if isinstance(keep_columns, str):
				keep_columns = [keep_columns]
		elif keep_columns:
			keep_columns = [col for col in X.columns if col != 'features']
		else:
			keep_columns = []

		transformed = assembled.select(*keep_columns, *[
			_get_distance('features', f.lit(cluster)).alias(f'{prefix}{cluster + 1}')
			for cluster in range(len(centers))
		])
		return transformed

	def _predict_spark(self, X, prefix='cluster_', keep_columns=False):
		"""
		:type X: SparkDF
		:rtype: SparkDF
		"""
		assembled = self._spark_assembler.transform(X)
		result = self._spark_model.transform(assembled)

		@f.udf(StringType())
		def _convert_prediction_to_string(x):
			return f'{prefix}{x + 1}'
		result = result.withColumn('prediction', _convert_prediction_to_string('prediction'))

		if keep_columns:
			if isinstance(keep_columns, str):
				keep_columns = [keep_columns]

			if isinstance(keep_columns, list):
				columns = keep_columns + ['prediction']
				return result.select(*columns)
			else:
				return result.drop('features')
		else:
			return result.select('prediction')

	################ GENERAL METHODS ###################

	def fit(self, X, y=None):
		"""
		:type X: SparkDF or PandasDF
		"""
		if isinstance(X, SparkDF):
			self._env = 'spark'
		elif isinstance(X, PandasDF):
			self._env = 'python'
		else:
			raise RuntimeError(f'unknown data type: {type(X)}')

		if self._evaluator is None:
			self._evaluator = ClusteringEvaluator()

		if self._env == 'spark':
			return self._fit_spark(X=X)

		elif self._env == 'python':
			return self._fit_python(X=X)

		else:
			raise RuntimeError(f'unknown environment {self._env}')

	@property
	def cluster_centers(self):
		"""
		:rtype: list[np_array]
		"""
		if self._env == 'spark':
			return self._spark_cluster_centers
		else:
			return self._python_cluster_centers

	def transform(self, X, keep_columns=False):
		"""
		:type X: SparkDF or PandasDF
		"""
		if self._env == 'spark':
			return self._transform_spark(X=X, keep_columns=keep_columns)

		elif self._env == 'python':
			return self._transform_python(X=X, keep_columns=keep_columns)

		else:
			raise RuntimeError(f'unknown environment {self._env}')

	def fit_transform(self, X, y=None, keep_columns=False):
		"""
		:type X: SparkDF or PandasDF
		"""
		self.fit(X=X, y=y)
		return self.transform(X=X, keep_columns=keep_columns)

	def predict(self, X, prefix='cluster_', keep_columns=False, return_dataframe=None):
		"""
		:type X: SparkDF or PandasDF or PandasSeries
		"""
		if self._env == 'spark':
			return self._predict_spark(X=X, prefix=prefix, keep_columns=keep_columns)

		elif self._env == 'python':
			return self._predict_python(
				X=X, prefix=prefix, keep_columns=keep_columns, return_dataframe=return_dataframe
			)

		else:
			raise RuntimeError(f'unknown environment {self._env}')

	def fit_predict(self, X, prefix='cluster_', keep_columns=False, y=None):
		self.fit(X=X, y=None)
		predictions = self.predict(X=X, prefix=prefix, keep_columns=keep_columns)
		return predictions

	def transform_and_predict(self, X, keep_columns=False):
		"""
		:type X: PandasDF or SparkDF
		:type keep_columns: bool or str or list[str]
		:rtype: PandasDF
		"""
		transformed = self.transform(X=X, keep_columns=keep_columns)
		predictions = self.predict(X=X, return_dataframe=False, keep_columns=False)

		if self._env == 'python':
			transformed['prediction'] = predictions

		elif self._env == 'spark':
			w = Window.orderBy(f.monotonically_increasing_id())
			transformed = transformed.withColumn('_index', f.row_number().over(w))
			predictions = predictions.withColumn('_index', f.row_number().over(w))
			transformed = transformed.join(predictions, on='_index', how='inner').drop('_index')

		else:
			raise RuntimeError(f'unknown environment {self._env}')

		return transformed

	@property
	def n_clusters(self):
		"""
		:rtype: int
		"""
		return self._n_clusters

	@property
	def timedout(self):
		return self._timed_out

	@property
	def distortion(self):
		self._model_dict.get('distortion', None)

	@property
	def inertia(self):
		return self._model_dict.get('inertia', None)

	@property
	def silhouette_score(self):
		return self._model_dict.get('silhouette_score', None)
