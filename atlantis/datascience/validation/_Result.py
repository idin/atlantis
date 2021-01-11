from pandas import DataFrame
from ..evaluation import evaluate_classification, evaluate_regression
from ..wrangling import bring_to_front
from ...exceptions import ComparisonError


class Result:
	def __init__(self, actual, predicted, probabilities, data_type, problem_type):
		self._actual = actual
		self._predicted = predicted
		self._probabilities = probabilities
		self._performance = None
		self._data_type = data_type
		self._problem_type = problem_type

	@property
	def performance(self):
		if self._performance is None:
			if self._problem_type.lower()[0] == 'r':
				self._performance = evaluate_regression(actual=self._actual, predicted=self._predicted)
			else:
				self._performance = evaluate_classification(actual=self._actual, predicted=self._predicted)
		return self._performance

	@property
	def data(self):
		if self._probabilities is None:
			result = DataFrame({
				'actual': self._actual,
				'predicted': self._predicted
			})
		else:
			result = DataFrame({
				'actual': self._actual,
				'predicted': self._predicted,
				'prediction_probability': self._probabilities
			})
		result['result_type'] = self._data_type
		return bring_to_front(result, columns=['result_type'])


class TestResult(Result):
	def __init__(self, actual, predicted, probabilities, problem_type):
		super().__init__(
			actual=actual, predicted=predicted, probabilities=probabilities, data_type='test',
			problem_type=problem_type
		)


class TrainingResult(Result):
	def __init__(self, actual, predicted, probabilities, problem_type):
		super().__init__(
			actual=actual, predicted=predicted, probabilities=probabilities, data_type='training',
			problem_type=problem_type
		)
