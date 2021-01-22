from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame, concat
from pandas.api.types import is_numeric_dtype
from .exceptions import NoColumnsError, ImputerNotFittedError
from ._find_good_columns import find_best_coverage_combination
ROW_NUM_COL = '_imputer_row_number_'


class SingleColumnImputer:
	def __init__(self, estimator, column, use_columns, column_type=None):
		"""
		:type estimator: LinearRegression or DecisionTreeClassifier or HyperModel
		:type column: str
		"""

		self._estimator = estimator
		self._imputed_column = column
		self._helper_columns = None
		self._coverage = None
		self._use_columns = use_columns
		self._fitted = None
		self._column_type = column_type
		if self._estimator is None:
			raise RuntimeError(f'no estimator provided!')

	def fit(self, X):
		"""
		:type X: DataFrame
		"""
		d = find_best_coverage_combination(data=X, column=self._imputed_column)
		self._helper_columns = [column for column in d['combination'] if column in self._use_columns]
		self._coverage = d

		data = X[self._helper_columns + [self._imputed_column]]

		training_data = data.dropna(axis=0)
		training_X = training_data[self._helper_columns]
		training_y = training_data[self._imputed_column]

		if training_X.shape[1] == 0:
			raise NoColumnsError('X has no columns!')
		elif training_X.shape[0] == 0:
			raise NoColumnsError('X has no rows!')

		if self._estimator is None:
			raise RuntimeError('no estimator')

		try:
			self._estimator.fit(training_X, training_y)
		except:
			display(training_X.head(), training_y.head())
			raise

		self._fitted = True
		if self._column_type is None:
			if is_numeric_dtype(training_y):
				self._column_type = 'numerical'
			else:
				self._column_type = 'nonnumerical'

	@property
	def type(self):
		if self._column_type.startswith('numeric'):
			return 'regressor'
		else:
			return 'classifier'

	def impute_column(self, data):
		missing = data[data[self._imputed_column].isna()].copy()
		not_missing = data[data[self._imputed_column].notna()]
		missing[self._imputed_column] = self._estimator.predict(missing[self._helper_columns])

		all_data = concat([missing, not_missing]).sort_values(ROW_NUM_COL)
		return all_data[self._imputed_column]

	def transform(self, X):
		"""
		:type X: DataFrame
		:rtype: DataFrame
		"""
		if not self._fitted:
			raise ImputerNotFittedError('imputer is not fitted yet!')

		data = X.copy()

		data[ROW_NUM_COL] = range(data.shape[0])
		data[self._imputed_column] = self.impute_column(data=data)
		return data.drop(columns=ROW_NUM_COL)

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)
