from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from pandas.api.types import is_numeric_dtype
from copy import deepcopy
from ...time.progress import iterate
from ...multiprocessing import SimpleController
from ...collections import get_intersection
from ._ColumnImputer import ColumnImputer


class DataImputer:
	def __init__(
			self, estimators, ignore_columns=None, helper_columns=None, impute_columns=None,
			cross_validation=None, drop_na=False, main_metric='rmse', impute_categorical=False,
			controller=None
	):
		"""
		:type estimators: LinearRegression or DecisionTreeClassifier or list
		:type ignore_columns: NoneType or str or list[str]
		:type helper_columns: NoneType or str or list[str]
		:type impute_columns: NoneType or str or list[str]
		:type cross_validation: CrossValidation or NoneType
		:type drop_na: bool
		:type main_metric: str
		:type impute_categorical: bool
		:type controller: BaseController
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
		self._drop_na = drop_na
		self._main_metric = main_metric.lower()
		self._impute_categorical = impute_categorical
		self._column_imputers = {}
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

	def _get_controller(self):
		"""
		:rtype: SimpleController or NoneType
		"""
		return self._controller

	def _get_estimators(self):
		return [deepcopy(estimator) for estimator in self._estimators]

	def fit(self, X, echo=1):
		ignore_columns = self._get_ignore_columns(data=X)
		helper_columns = self._get_helper_columns(data=X, ignore_columns=ignore_columns)
		impute_columns = self._get_impute_columns(data=X, ignore_columns=ignore_columns)

		def _create_column_imputer(column):
			column_imputer = ColumnImputer(
				column=column, estimators=self._get_estimators(), helper_columns=helper_columns,
				cross_validation=self._cross_validation, drop_na=self._drop_na,
				main_metric=self._main_metric
			)
			column_imputer.fit(X=X)
			return column_imputer

		if self._controller is None:
			self._column_imputers = {
				column: _create_column_imputer(column=column)
				for column in iterate(impute_columns)
			}

		else:
			controller = self._get_controller()
			for column in impute_columns:
				controller.add_task(
					function=_create_column_imputer, kwargs={'column': column},
					task_id=f'{column}_imputer'
				)
			controller.do(echo=echo)
			self._column_imputers = {
				column: controller.get_result(task_id=f'{column}_imputer')
				for column in impute_columns
			}

	def transform(self, X, echo=1):
		X = X.copy()

		def _impute_column(imputer):
			"""
			:type imputer: ColumnImputer
			:rtype: list
			"""
			return {'name': imputer._imputed_column, 'values': imputer._transform(X=X, return_y=True)}

		if self._controller is None:
			column_names_and_values = [
				_impute_column(imputer=imputer)
				for imputer in iterate(self._column_imputers.values())
			]

		else:
			controller = self._get_controller()
			for imputer in self._column_imputers.values():
				controller.add_task(
					function=_impute_column, kwargs={'imputer': imputer},
					task_id=f'{imputer._imputed_column}_values'
				)
			controller.do(echo=echo)
			column_names_and_values = [
				controller.get_result(task_id=f'{column}_values')
				for column in self._column_imputers.keys()
			]

		for d in column_names_and_values:
			name = d['name']
			values = d['values']
			X[name] = values

		return X
