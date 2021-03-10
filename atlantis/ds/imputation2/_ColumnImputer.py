from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from pandas import concat
from ..validation import CrossValidation
from ._fill_na_and_flag import NaFillerFlagger
ROW_NUM_COLUMN = '_imputer_row_number_'


class ColumnImputer:
	def __init__(
			self, column, estimators, helper_columns=None, cross_validation=None,
			drop_na=False,
			main_metric='rmse'
	):
		"""
		:type estimator: LinearRegression or DecisionTreeClassifier or list
		:type column: str
		:type helper_columns: list[str] or NoneType
		:type cross_validation: CrossValidation or NoneType
		"""
		self._imputed_column = column
		self._helper_columns = helper_columns
		self._estimators = estimators
		self._cross_validation = cross_validation
		self._cross_validation_evaluation = None
		self._drop_na = drop_na
		self._main_metric = main_metric.lower()
		self._na_filler = None

		self._x_columns = None

	def fit(self, X):
		if self._imputed_column not in X.columns:
			raise KeyError(f'column "{self._imputed_column}" does not exist in the data!')
		if self._helper_columns is not None:
			missing_columns = [f'"{col}"' for col in self._helper_columns if col not in X.columns]
			if len(missing_columns) > 0:
				raise KeyError(f'missing columns: {", ".join(missing_columns)}')
		else:
			self._helper_columns = [col for col in X.columns if col != self._imputed_column]

		data = X[self._helper_columns + [self._imputed_column]]

		if self._drop_na:
			training_data = data.dropna(axis=0)
			training_y = training_data[self._imputed_column]
			training_x = training_data[self._helper_columns]

		else:
			training_data = data[data[self._imputed_column].notna()]

			training_y = training_data[self._imputed_column]
			training_x_with_missing = training_data[self._helper_columns]
			self._na_filler = NaFillerFlagger()
			training_x = self._na_filler.fit_transform(X=training_x_with_missing)

		if training_x.shape[1] == 0:
			raise RuntimeError('X has no columns!')
		elif training_x.shape[0] == 0:
			raise RuntimeError('X has no rows!')

		if self._estimators is None:
			raise RuntimeError('no estimator')

		if self._cross_validation is not None:
			training_data = training_x.copy()
			training_data[[self._imputed_column]] = training_y
			validation_container = self._cross_validation.split(data=training_data)
			self._cross_validation_evaluation = validation_container.evaluate_regression(
				estimators=self._estimators, x_columns=None, y_column=self._imputed_column
			)

		self._estimator.fit(training_x, training_y)
		self._x_columns = training_x.columns

	def transform(self, X):
		data = X.copy()
		data[ROW_NUM_COLUMN] = range(data.shape[0])

		missing = data[data[self._imputed_column].isna()].copy()
		not_missing = data[data[self._imputed_column].notna()]

		missing_x = missing[self._helper_columns]
		if self._na_filler is not None:
			missing_x = self._na_filler.transform(missing_x)
			if missing_x.isnull().values.any():
				raise RuntimeError('transform did not remove missing values')
		try:
			if missing.shape[0] > 0:
				missing[self._imputed_column] = self._estimator.predict(missing_x)
		except Exception as e:
			display(missing_x)
			raise e

		if missing.shape[0] > 0:
			all_data = concat([missing, not_missing]).sort_values(ROW_NUM_COLUMN)
		else:
			all_data = not_missing.sort_values(ROW_NUM_COLUMN)

		return all_data.drop(columns=ROW_NUM_COLUMN)
