

from pandas.api.types import is_numeric_dtype
from ...collections import get_intersection
from ...multiprocessing import JobController


class Imputer:
	def __init__(
			self, estimators, cross_validation, ignore_columns=None, helper_columns=None, impute_columns=None,
			main_metric='rmse', impute_categorical=False,
			controller=None
	):
		"""
		:type estimators: LinearRegression or DecisionTreeClassifier or list
		:type ignore_columns: NoneType or str or list[str]
		:type helper_columns: NoneType or str or list[str]
		:type impute_columns: NoneType or str or list[str]
		:type cross_validation: CrossValidation or NoneType
		:type main_metric: str
		:type impute_categorical: bool
		"""
		if ignore_columns is None:
			ignore_columns = []
		elif isinstance(ignore_columns, str):
			ignore_columns = [ignore_columns]
		elif not isinstance(ignore_columns, list):
			raise TypeError(f'ignore_columns cannot be of type {type(ignore_columns)}')

		self._ignore_columns = ignore_columns

		if helper_columns is not None:
			if isinstance(helper_columns, str):
				helper_columns = [helper_columns]

			if len(get_intersection(helper_columns, ignore_columns)) > 0:
				raise ValueError(f'helper_columns and ignore_columns intersect: {get_intersection(helper_columns, ignore_columns)}')

		self._helper_columns = helper_columns

		if impute_columns is not None:
			if isinstance(impute_columns, str):
				impute_columns = [impute_columns]

			if len(get_intersection(impute_columns, ignore_columns)) > 0:
				raise ValueError(f'impute_columns and ignorre_columns intersect: {get_intersection(impute_columns, ignore_columns)}')

		self._impute_columns = impute_columns

		if not isinstance(estimators, (list, tuple)):
			estimators = [estimators]

		self._estimators = estimators
		self._cross_validation = cross_validation
		self._main_metric = main_metric.lower()
		self._impute_categorical = impute_categorical
		self._column_imputers = {}

		if controller is None:
			controller = JobController()
		self._controller = controller

	def _get_ignore_columns(self, data):
		return [col for col in data.columns if col in self._ignore_columns]

	def _get_helper_columns(self, data, ignore_columns):
		if self._helper_columns is not None:
			return self._helper_columns
		else:
			return [col for col in data.columns if col not in ignore_columns]

	def _get_impute_columns(self, data, ignore_columns):
		if self._impute_columns is not None:
			return self._impute_columns
		else:
			impute_columns = [
				col for col in data.columns
				if col not in ignore_columns and data[col].isnull().values.any()
			]
			if self._impute_categorical:
				return impute_columns
			else:
				return [col for col in impute_columns if is_numeric_dtype(data[col])]

	def fit(self, X, echo=1):
		ignore_columns = self._get_ignore_columns(data=X)
		helper_columns = self._get_helper_columns(data=X, ignore_columns=ignore_columns)
		impute_columns = self._get_impute_columns(data=X, ignore_columns=ignore_columns)

		def _evaluate_estimator(impute_column, fold_num, )


		for