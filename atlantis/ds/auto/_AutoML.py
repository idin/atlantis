from ..parallel_computing import Processor
from ..parallel_computing import CrossValidationProject
from ..validation import CrossValidation, TimeSeriesValidation, Scoreboard
from ..validation import EstimatorRepository
import multiprocess
from pandas import DataFrame


class AutoML:
	def __init__(self):
		"""

		"""
		self._processor = Processor()
		self._estimator_repositories = {}

	@property
	def projects(self):
		"""
		:rtype: dict[str, CrossValidationProject]
		"""
		return self.processor.projects

	@staticmethod
	def get_cpu_count():
		return multiprocess.cpu_count()

	def get_worker_count(self):
		return self.processor.get_worker_count()

	@property
	def processor(self):
		"""
		:rtype: Processor
		"""
		return self._processor

	@property
	def estimator_repositories(self):
		"""
		:rtype: dict[str, EstimatorRepository]
		"""
		return self._estimator_repositories

	@property
	def score_boards(self):
		"""
		:rtype: dict[str, Scoreboard]
		"""
		return self._score_boards

	def add_validation(self, project_name, problem_type, y_column, data, validation, main_metric=None, random_state=None):
		"""
		:param 	project_name: the name or identification for a problem, should be unique
		:type	project_name: str or int

		:param 	problem_type: regression or classification
		:type 	problem_type: str

		:param 	y_column: name of the y column
		:type 	y_column: str

		:param	data: data used for the problem
		:type	data: DataFrame

		:param	validation: a cross validation or time-series validation skeleton
		:type 	validation: CrossValidation or TimeSeriesValidation

		:type 	random_state: int or NoneType
		"""

		self._projects[project_name] = CrossValidationProject(
			name=project_name, y_column=y_column, problem_type=problem_type, time_unit='ms', evaluation_function=None,
			main_metric=main_metric
		)

		validation_container = validation.split(data=data, random_state=random_state)

		estimator_repository = EstimatorRepository()
		self._estimator_repositories[project_name] = estimator_repository

		# add validation as a new problem
		validation_problem_id = f'{project_name}_validation'

		self._score_boards[validation_problem_id] = Scoreboard(
			estimator_repository=estimator_repository,
			main_metric=main_metric, lowest_is_best=lowest_is_best, best_score=best_score
		)

		self.processor.add_project(
			project_name=validation_problem_id, problem_type=problem_type, y_column=y_column
		)

		for i, fold in enumerate(validation_container.folds):
			data_id = f'fold_{i + 1}'
			self.processor.add_data(
				project_name=validation_problem_id,
				data_id=data_id,
				training_data=fold.training_data,
				test_data=fold.test_data
			)
			self.score_boards[project_name].add_training_test_id(training_test_id=data_id)

		# add final test as a new problem
		final_test_problem_id = f'{project_name}_final_test'

		self.processor.add_project(
			project_name=final_test_problem_id, problem_type=problem_type, y_column=y_column
		)

		self.processor.add_data(
			project_name=final_test_problem_id, data_id=f'final_test',
			training_data=validation_container.validation,
			test_data=validation_container.holdout
		)

	def add_models(self, problem_id, estimator, kwargs):
		self.estimator_repositories[problem_id].append(estimator=estimator, estimator_arguments=kwargs)

	def add_workers(self, num_workers=1):
		self.processor.add_workers(num_workers=num_workers)




