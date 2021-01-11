from sklearn.linear_model import LinearRegression, LogisticRegression
from copy import deepcopy
from sklearn.base import is_classifier, is_regressor
from ._Result import TrainingResult, TestResult
from ...exceptions import EstimatorNotEvaluatedError, ComparisonError
from ...time.time import Job


class Evaluation(Job):
	def __init__(self, estimator, training_x, training_y, test_x, test_y, estimator_id, problem_type, main_metric):
		super().__init__()
		self._training_result = None
		self._test_result = None
		self._estimator = estimator
		self._estimator_id = estimator_id
		self._training_x = training_x
		self._training_y = training_y
		self._test_x = test_x
		self._test_y = test_y
		self._evaluated = False

		if problem_type is None:
			if is_regressor(estimator):
				problem_type = 'regression'
			elif is_classifier(estimator):
				problem_type = 'classification'
			else:
				raise ValueError('problem type is not provided!')

		if problem_type.lower().startswith('regres'):
			problem_type = 'regression'
		elif problem_type.lower().startswith('class'):
			problem_type = 'classification'

		if main_metric is None:
			if problem_type == 'regression':
				main_metric = 'smape'
			elif problem_type == 'classification':
				main_metric = 'f1_score'
			else:
				raise ValueError('main metric should be provided!')

		self._problem_type = problem_type
		self._main_metric = main_metric

	@property
	def estimator_name(self):
		return str(self._estimator.__class__.__name__)

	@property
	def estimator_id(self):
		return self._estimator_id

	def fit(self):
		"""
		:rtype: LinearRegression or LogisticRegression
		"""
		estimator = deepcopy(self._estimator)
		estimator.fit(X=self._training_x, y=self._training_y)
		return estimator

	def evaluate(self, estimator):
		"""
		:type estimator: LinearRegression or LogisticRegression
		"""
		test_predicted = estimator.predict(self._test_x)
		test_actual = self._test_y

		training_predicted = estimator.predict(self._training_x)
		training_actual = self._training_y

		try:
			test_probabilities = estimator.predict_proba(self._test_x)
			training_probabilities = estimator.predict_proba(self._training_x)
		except AttributeError:
			test_probabilities = None
			training_probabilities = None

		self._test_result = TestResult(
			actual=test_actual, predicted=test_predicted, probabilities=test_probabilities,
			problem_type=self._problem_type
		)
		self._training_result = TrainingResult(
			actual=training_actual, predicted=training_predicted, probabilities=training_probabilities,
			problem_type=self._problem_type
		)
		self._evaluated = True

	def do(self):
		"""
		:rtype: LinearRegression or LogisticRegression
		"""
		self.start()
		estimator = self.fit()
		self.evaluate(estimator=estimator)
		self.end()
		return estimator

	def is_evaluated(self):
		return self._evaluated

	@property
	def training(self):
		"""
		:rtype: TrainingResult
		"""
		if self.is_evaluated():
			return self._training_result
		else:
			raise EstimatorNotEvaluatedError(f'estimator "{self.estimator_name}" is not evaluated yet!')

	@property
	def test(self):
		"""
		:rtype: TestResult
		"""
		if self.is_evaluated():
			return self._test_result
		else:
			raise EstimatorNotEvaluatedError(f'estimator "{self.estimator_name}" is not evaluated yet!')

	@property
	def performance(self):
		return self.test.performance[self._main_metric]

	def _check_comparability(self, other):
		"""
		:type other: Evaluation
		"""
		if not isinstance(other, self.__class__):
			raise TypeError(f'{other} is of type {type(other)}!')
		if self._problem_type != other._problem_type:
			raise ComparisonError(
				f'Cannot compare a problem type of "{self._problem_type}" with "{other._problem_type}"!'
			)
		if self._main_metric != other._main_metric:
			raise ComparisonError(
				f'Main metric is ambiguous!'
			)

	def __gt__(self, other):
		"""
		:type other: Evaluation
		"""
		self._check_comparability(other)
		if self._problem_type.lower()[0] == 'r':  #  regression:
			return self.performance < other.performance
		else:
			return self.performance > other.performance

	def __eq__(self, other):
		"""
		:type other: Evaluation
		"""
		self._check_comparability(other)
		return self.performance == other.performance

	def __ge__(self, other):
		return self == other or self > other

	def __lt__(self, other):
		return other > self

	def __le__(self, other):
		return other >= self

	def __ne__(self, other):
		return not (self == other)
