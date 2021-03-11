
from pandas import DataFrame
from ...exceptions import MissingArgumentError
from ._Evaluation import Evaluation
from ._DataContainer import get_display_function, DataContainer


class TrainingTestContainer(DataContainer):
	def __init__(self, data, training_indices, test_indices, x_columns=None, y_column=None):
		super().__init__(data=data, x_columns=x_columns, y_column=y_column)
		self._training_indices = training_indices
		self._test_indices = test_indices

	@property
	def training_indices(self):
		return self._training_indices

	@property
	def test_indices(self):
		return self._test_indices

	@property
	def training_data(self):
		"""
		:rtype: DataFrame
		"""
		return self._data.iloc[self._training_indices]

	@property
	def test_data(self):
		"""
		:rtype: DataFrame
		"""
		return self._data.iloc[self._test_indices]

	def display(self, p=None, prefix='', function=None):
		if function is None:
			display = get_display_function()
		else:
			display = function

		if display is False:
			print(f'{prefix}Training:')
			print(self.training_data)
			print(f'{prefix}Test:')
			print(self.test_data)
		else:
			print(f'{prefix}Training:')
			display(self.training_data)
			print(f'{prefix}Test:')
			display(self.test_data)

	def fit(self, model, x_columns=None, y_column=None):
		"""
		:type model: LinearRegression or LogisticRegression
		:type x_columns: list[str]
		:type y_column: str
		"""
		x_columns = x_columns or self._x_columns
		y_column = y_column or self._y_column
		if y_column is None:
			raise MissingArgumentError('y_column should be provided!')
		elif x_columns is None:
			x_columns = [col for col in self._data.columns if col != y_column]

		model.fit(self.training_data[x_columns], self.training_data[y_column])
		return model

	def get_evaluation(self, estimator, estimator_id, x_columns=None, y_column=None, problem_type=None, main_metric=None):
		"""
		:type estimator: LinearRegression or LogisticRegression
		:type estimator_id: int or str or tuple
		:type x_columns: list[str]
		:type y_column: str
		:type problem_type: str or NoneType
		:type main_metric: str or NoneType
		"""
		x_columns = x_columns or self._x_columns
		y_column = y_column or self._y_column
		if y_column is None:
			raise MissingArgumentError('y_column should be provided!')
		elif x_columns is None:
			x_columns = [col for col in self._data.columns if col != y_column]

		evaluation = Evaluation(
			estimator=estimator, estimator_id=estimator_id,
			training_x=self.training_data[x_columns], training_y=self.training_data[y_column],
			test_x=self.test_data[x_columns], test_y=self.test_data[y_column],
			problem_type=problem_type, main_metric=main_metric
		)
		return evaluation


