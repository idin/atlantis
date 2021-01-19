from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame, concat
from pandas.api.types import is_numeric_dtype
from .exceptions import NoColumnsError, ImputerNotFittedError
ROW_NUM_COL = '_imputator_row_number_'


class SingleColumnImputer:
	def __init__(self, model, column, helper_columns=None, column_type=None):
		"""
		:type model: LinearRegression or DecisionTreeClassifier or HyperModel
		:type column: str
		"""

		self._model = model
		self._imputed_column = column
		self._helper_columns = helper_columns
		self._fitted = None
		self._column_type = column_type

	def fit(self, X):
		"""
		:type X: DataFrame
		"""
		y = X[self._imputed_column]
		X = X.drop(columns=self._imputed_column)
		if self._helper_columns is None:
			self._helper_columns = list(X.columns)
		else:
			X = X[self._helper_columns]

		if X.shape[1] == 0:
			raise NoColumnsError('X has no columns!')
		elif X.shape[0] == 0:
			raise NoColumnsError('X has no rows!')

		X = X[y.notna()]
		y = y[y.notna()]

		self._model.fit(X, y)
		self._fitted = True
		if self._column_type is None:
			if is_numeric_dtype(y):
				self._column_type = 'numerical'
			else:
				self._column_type = 'nonnumerical'

	@property
	def type(self):
		if self._column_type.startswith('numeric'):
			return 'regressor'
		else:
			return 'classifier'

	def imputate_column(self, data):
		missing = data[data[self._imputed_column].isna()].copy()
		not_missing = data[data[self._imputed_column].notna()]
		missing[self._imputed_column] = self._model.predict(missing[self._helper_columns])

		all_data = concat([missing, not_missing]).sort_values(ROW_NUM_COL)
		return all_data[self._imputed_column]

	def transform(self, X):
		"""
		:type X: DataFrame
		:rtype: DataFrame
		"""
		if not self._fitted:
			raise ImputerNotFittedError('imputator is not fitted yet!')

		data = X.copy()

		data[ROW_NUM_COL] = range(data.shape[0])
		self.imputate_column(data=data)
		return data.drop(columns=ROW_NUM_COL)

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)
