from ._Task import TrainingTestTask
from ._TimeEstimate import TimeEstimate, MissingTimeEstimate
from ..evaluation import evaluate_regression, evaluate_classification
from ..validation import Scoreboard


class Project:
	def __init__(self, name, y_column, problem_type, time_unit='ms', evaluation_function=None):
		self._name = name

		if problem_type.lower().startswith('reg'):
			problem_type = 'regression'
		elif problem_type.lower().startswith('class'):
			problem_type = 'classification'
		else:
			raise ValueError(f'problem_type: {problem_type} is not defined!')

		self._problem_type = problem_type

		if evaluation_function is None:
			if problem_type == 'regression':
				evaluation_function = evaluate_regression
			elif problem_type == 'classification':
				evaluation_function = evaluate_classification

		self._evaluation_function = evaluation_function
		self._time_estimates = {}
		self._y_column = y_column
		self._data_ids = set()
		self._time_unit = time_unit
		self._estimators = {}
		self._scoreboard = None

	def add_scoreboard(self, main_metric=None, lowest_is_best=None, best_score=None):
		if self.problem_type == 'regression':
			main_metric = 'rmse' if main_metric is None else main_metric
			lowest_is_best = True if lowest_is_best is None else lowest_is_best
			best_score = 0 if best_score is None else best_score
		elif self.problem_type == 'classification':
			main_metric = 'f1_score' if main_metric is None else main_metric
			lowest_is_best = False if lowest_is_best is None else lowest_is_best
			best_score = 1 if best_score is None else best_score
		else:
			raise ValueError(f'problem type {self.problem_type} does not work with validation!')
		self._scoreboard = Scoreboard(main_metric=main_metric, lowest_is_best=lowest_is_best, best_score=best_score)
		for data_id in self.data_ids:
			self._scoreboard.add_data_id(data_id=data_id)
		for estimator_type, estimator_id in self._estimators.keys():
			self._scoreboard.add_estimator(estimator_type=estimator_type, estimator_id=estimator_id)

	@property
	def problem_type(self):
		return self._problem_type

	@property
	def name(self):
		return self._name

	def __repr__(self):
		return f'Problem {self.name}'

	def __str__(self):
		return repr(self)

	@property
	def evaluation_function(self):
		"""
		:rtype: callable
		"""
		return self._evaluation_function

	@property
	def y_column(self):
		"""
		:rtype: str
		"""
		return self._y_column

	@property
	def time_estimates(self):
		"""
		:rtype: dict[str, TimeEstimate]
		"""
		return self._time_estimates

	def add_time_estimate(self, task):
		"""
		:type task: TrainingTestTask
		"""
		if task.estimator_type not in self.time_estimates:
			self.time_estimates[task.estimator_type] = TimeEstimate()
		self.time_estimates[task.estimator_type].append(task.get_elapsed(unit=self._time_unit))

	def get_time_estimate(self, task):
		if task.project_name != self.name:
			raise RuntimeError(f'task.problem_id = {task.project_name} does not match problem_id = {self.name}')

		if task.is_done():
			return task.get_elapsed(unit=self._time_unit)

		if task.estimator_type in self.time_estimates:
			return self.time_estimates[task.estimator_type].get_mean()

		elif len(self.time_estimates) > 0:
			total = 0
			count = 0
			for estimate in self.time_estimates.values():
				total += estimate.mean
				count += 1
			return total / count

		else:
			return MissingTimeEstimate()

	@property
	def scoreboard(self):
		"""
		:rtype: Scoreboard
		"""
		return self._scoreboard

	def add_data_id(self, data_id):
		if data_id in self._data_ids:
			raise ValueError(f'data {data_id} already exists for {self}!')
		self._data_ids.add(data_id)
		if self._scoreboard is not None:
			self.scoreboard.add_data_id(data_id=data_id)

	@property
	def data_ids(self):
		"""
		:rtype: set
		"""
		return self._data_ids

	def add_estimator(self, estimator_class, estimator_id, kwargs):
		if not isinstance(estimator_class, type):
			raise TypeError(f'estimator_class is of type {type(estimator_class)}')
		estimator_type = estimator_class.__name__  # string
		key = estimator_type, estimator_id
		self._estimators[key] = {'class': estimator_class, 'kwargs': kwargs}
		if self._scoreboard is not None:
			self.scoreboard.add_estimator(estimator_type=estimator_type, estimator_id=estimator_id)

	def get_all_estimator_data_combinations(self):
		"""
		:rtype: list[dict]
		"""
		return [
			{
				'estimator_type': estimator_type_and_id[0],
				'estimator_id': estimator_type_and_id[1],
				'estimator_class': estimator_class_and_kwargs[0],
				'estimator_kwargs': estimator_class_and_kwargs[1],
				'data_id': data_id
			}
			for data_id in self.data_ids
			for estimator_type_and_id, estimator_class_and_kwargs in self._estimators.items()
		]
