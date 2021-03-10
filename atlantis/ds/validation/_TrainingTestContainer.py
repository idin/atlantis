
from pandas import DataFrame
from ...exceptions import MissingArgumentError
from ._Evaluation import Evaluation
from ..evaluation import evaluate_regression, evaluate_classification
from ._DataContainer import get_display_function, DataContainer
from ...time.progress import ProgressBar


class TrainingTestContainer(DataContainer):
	def __init__(
			self, data, training_indices, test_indices, x_columns=None, y_column=None,
			sort_columns=None
	):
		super().__init__(data=data, x_columns=x_columns, y_column=y_column, sort_columns=sort_columns)
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

	def _evaluate_estimator(self, estimator, evaluation_type, x_columns=None, y_column=None):
		y_column = y_column or self._y_column
		if y_column is None:
			raise ValueError(f'y_column should be provided')
		x_columns = x_columns or self._x_columns
		if x_columns is None:
			x_columns = [col for col in self.columns if col != y_column]


	def _evaluate(self, estimators, evaluation_type, x_columns=None, y_column=None, echo=1):
		if isinstance(estimators, (list, tuple)):
			estimators = {(i + 1): x for i, x in enumerate(estimators)}

		if isinstance(estimators, dict):
			records = []
			progress_bar = ProgressBar(total=len(estimators), echo=echo)
			progress_amount = 0
			for estimator_id, estimator in estimators.items():
				progress_bar.show(amount=progress_amount, text=f'evaluating estimator {estimator_id}')
				evaluation = self._evaluate_estimator(
					estimator=estimator, evaluation_type=evaluation_type,
					x_columns=x_columns, y_column=y_column
				)
				records.append({
					'estimator_id': estimator_id,
					'estimator_name': estimator.__class__.__name__,
					**evaluation
				})
				progress_amount += 1
			progress_bar.show(amount=progress_amount, text=f'evaluation complete!')
			return DataFrame.from_records(records)

		else:
			return self._evaluate_estimator(
				estimator=estimators, evaluation_type=evaluation_type,
				x_columns=x_columns, y_column=y_column
			)

	def evaluate_regression(self, estimators, x_columns=None, y_column=None, echo=1):
		"""
		evaluates one regressor or more and returns a dictionary or a dataframe of the results
		:param estimators: one or more regressors
		:param x_columns: the x columns (independent variables)
		:param y_column: the y column (dependent variable)
		:param echo: to show progress or not
		:rtype: dict or DataFrame
		"""
		return self._evaluate(
			estimators=estimators, evaluation_type='regression', x_columns=x_columns, y_column=y_column,
			echo=echo
		)

	def evaluate_classification(self, estimators, x_columns=None, y_column=None, echo=1):
		"""
		evaluates one classifier or more and returns a dictionary or a dataframe of the results
		:param estimators: one or more classifiers
		:param x_columns: the x columns (independent variables)
		:param y_column: the y column (dependent variable)
		:param echo: to show progress or not
		:rtype: dict or DataFrame
		"""
		return self._evaluate(
			estimators=estimators, evaluation_type='classification', x_columns=x_columns, y_column=y_column,
			echo=echo
		)

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


