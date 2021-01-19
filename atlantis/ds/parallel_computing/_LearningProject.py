from ..evaluation import evaluate_regression, evaluate_classification
from ..validation import Scoreboard, EstimatorRepository, TrainingTestContainer, Validation, TrainingTest
from ._Project import Project
from ._LearningTask import LearningTask, get_training_data_id, get_test_data_id
from pandas import DataFrame
from numpy import random


class LearningProject(Project):
	def __init__(
			self, name, y_column, problem_type, x_columns=None,
			time_unit='ms',
			evaluation_function=None, main_metric=None, lowest_is_best=None, best_score=None,
			scoreboard=None, processor=None
	):
		"""

		:type 	name: str
		:type 	y_column: strx

		:type 	problem_type: str
		:param 	problem_type: either regression or classification

		:type 	x_columns: str

		:type 	time_unit: str
		:param 	time_unit: s, ms, etc.


		:type 	evaluation_function: callable
		:param 	evaluation_function: 	a function that gets predicted and actual and
										produces a dictionary of values such as {'rmse': ...} for regression
										or {'f1_score': ...} for classification

		:param 	main_metric: 	the main metric used for comparison, it should exist as one of the keys
								in the result produced by evaluation_function

		:param 	lowest_is_best: usually True for regression (unless a weird metric is used) and False for classification
		:param 	best_score: usually 0 for regression and 1 for classification

		:type 	scoreboard: Scoreboard
		:param 	scoreboard: a Scoreboard object that keeps score of all estimators, can be added later too
		"""

		super().__init__(name=name, time_unit=time_unit, processor=processor)
		self._data_id_prefixes = set()
		if problem_type.lower().startswith('reg'):
			self._problem_type = 'regression'

		elif problem_type.lower().startswith('class'):
			self._problem_type = 'classification'

		else:
			raise ValueError(f'problem_type: {problem_type} is not defined!')

		if evaluation_function is None:
			if self.problem_type == 'regression':
				self._evaluation_function = evaluate_regression
				main_metric = 'rmse' if main_metric is None else main_metric
				lowest_is_best = True if lowest_is_best is None else lowest_is_best
				best_score = 0 if best_score is None else best_score

			elif self.problem_type == 'classification':
				self._evaluation_function = evaluate_classification
				main_metric = 'f1_score' if main_metric is None else main_metric
				lowest_is_best = False if lowest_is_best is None else lowest_is_best
				best_score = 1 if best_score is None else best_score
		else:
			self._evaluation_function = evaluation_function
			if main_metric is None or lowest_is_best is None or best_score is None:
				raise ValueError('main_metric, lowest_is_best, and best_score should be provided for evaluation_function')

		if y_column is None:
			raise ValueError('y_column should be provided')
		self._y_column = y_column
		self._x_columns = None
		if x_columns is not None:
			self.x_columns = x_columns

		if scoreboard is None:
			scoreboard = Scoreboard(main_metric=main_metric, lowest_is_best=lowest_is_best, best_score=best_score)
		self._scoreboard = scoreboard

	@property
	def problem_type(self):
		return self._problem_type

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
	def x_columns(self):
		"""
		:rtype list[str]
		"""
		return self._x_columns

	@x_columns.setter
	def x_columns(self, x_columns):
		if x_columns is None:
			raise ValueError('x_columns should be a list of strings!')
		if self.y_column in x_columns:
			raise KeyError(f'y_column "{self.y_column}" is among x_columns')
		self._x_columns = x_columns

	@property
	def scoreboard(self):
		"""
		:rtype: Scoreboard
		"""
		return self._scoreboard

	def _add_training_test_container(self, data_id_prefix, container, processor=None, overwrite=False):
		"""
		:type processor: atlantis.ds.parallel_computing.Processor
		:type data_id_prefix: str
		:type container: TrainingTestContainer
		:type overwrite: bool
		"""
		if processor is not None:
			processor.add_project(project=self)

		self.add_data_pair(
			processor=None, data_id_prefix=data_id_prefix,
			training_data=container.training_data, test_data=container.test_data,
			overwrite=overwrite
		)

	def add_training_test(
			self, data, training_test, data_id_prefix, processor=None, overwrite=False, random_state=None
	):
		"""

		:param data:
		:param training_test:
		:type  training_test: TrainingTest
		:param data_id_prefix:
		:param processor:
		:param overwrite:
		:param random_state:
		:return:
		"""
		if processor is not None:
			processor.add_project(project=self)

		container = training_test.split(data=data, random_state=random_state)
		self._add_training_test_container(
			processor=None, container=container,
			data_id_prefix=data_id_prefix, overwrite=overwrite
		)

	def add_validation(
			self, data, validation, processor=None, data_id_prefix='fold_',
			overwrite=False, random_state=None
	):
		"""

		:param data:
		:type  data: DataFrame

		:param validation:
		:type  validation: Validation

		:param processor:

		:param data_id_prefix:
		:type  data_id_prefix: str

		:type  overwrite: bool
		:type  random_state: int

		:rtype: LearningProject
		"""
		if processor is not None:
			processor.add_project(project=self)

		container = validation.split(data=data, random_state=random_state)
		for i, fold in enumerate(container.folds):
			data_id = f'{data_id_prefix}{i + 1}'
			self._add_training_test_container(
				processor=None, data_id_prefix=data_id, container=fold, overwrite=overwrite
			)
		return self

	def add_data(self, data_id_prefix, data_type, processor=None, data=None, overwrite=False):
		"""

		:type 	processor: atlantis.ds.parallel_computing.Processor

		:param 	data_id_prefix: cannot end with _training ot _test
		:type 	data_id_prefix: str or tuple or int

		:param 	data_type: either training or test
		:type 	data: DataFrame
		:param 	overwrite: if True, allows overwriting the data
		"""
		if processor is not None:
			processor.add_project(project=self)
		else:
			processor = self._processor

		if isinstance(data_id_prefix, str):
			if data_id_prefix.endswith('_training') or data_id_prefix.endswith('_test'):
				raise ValueError('data_id cannot contain training or test in it')
		else:
			raise TypeError(f'data_id_prefix should be a str but it is of type {type(data_id_prefix)}')

		if data_type == 'training':
			data_id = get_training_data_id(data_id_prefix=data_id_prefix)

		elif data_type == 'test':
			data_id = get_test_data_id(data_id_prefix=data_id_prefix)

		else:
			raise ValueError('data_type can either be training or test')

		if data is None:
			all_columns = processor.get_data_columns(data_id=data_id_prefix)
		else:
			all_columns = data.columns

		if self.y_column not in all_columns:
			raise KeyError(f'y_column "{self.y_column}" does not exist in data {data_id}')

		if self._x_columns is None:
			self._x_columns = [column for column in all_columns if column != self.y_column]
		else:
			missing_columns = [column for column in self.x_columns if column not in all_columns]
			if len(missing_columns) > 0:
				raise KeyError(f'columns missing: {missing_columns}')

		super().add_data(processor=None, data_id=data_id, data=data, overwrite=overwrite)

		if data_type == 'test':
			self.scoreboard.add_data_id(data_id=data_id_prefix)

	def add_data_pair(self, data_id_prefix, processor=None, training_data=None, test_data=None, overwrite=False):
		"""

		:type 	processor: atlantis.ds.parallel_computing.Processor
		:type 	data_id_prefix: str
		:type 	training_data: DataFrame
		:type 	test_data: DataFrame
		:param 	overwrite: if True, allows overwriting the data, otherwise, raises an error if data_id exists

		"""
		if processor is not None:
			processor.add_project(project=self)

		self.add_data(
			processor=None, data_id_prefix=data_id_prefix,
			data_type='training', data=training_data, overwrite=overwrite
		)
		self.add_data(
			processor=None, data_id_prefix=data_id_prefix,
			data_type='test', data=test_data, overwrite=overwrite
		)
		self._data_id_prefixes.add(data_id_prefix)

	def add_estimator(self, estimator_class, estimator_id, estimator_arguments):
		estimator_name, estimator_id = super().add_estimator(
			estimator_class=estimator_class, estimator_id=estimator_id, estimator_arguments=estimator_arguments
		)
		self.scoreboard.add_estimator(
			estimator_name=estimator_name, estimator_id=estimator_id
		)
		return estimator_name, estimator_id

	def add_estimator_repository(self, repository):
		"""
		:type repository: EstimatorRepository
		"""
		for dictionary in repository.estimator_dictionaries:
			estimator = dictionary['estimator']
			estimator_arguments = dictionary['estimator_arguments']
			estimator_id = dictionary['id']
			self.add_estimator(
				estimator_class=estimator, estimator_id=estimator_id, estimator_arguments=estimator_arguments
			)

	def produce_task(
			self, estimator_name, estimator_id, estimator_class, estimator_arguments, data_id_prefix,
			ignore_error=False
	):
		"""

		:param estimator_name:
		:param estimator_id:
		:param estimator_class:
		:param estimator_arguments:
		:param data_id_prefix:
		:rtype: LearningTask
		"""
		task = LearningTask(
			project_name=self.name, estimator_class=estimator_class,
			estimator_name=estimator_name, estimator_id=estimator_id,
			estimator_arguments=estimator_arguments,
			data_id_prefix=data_id_prefix, y_column=self.y_column, x_columns=self.x_columns,
			evaluation_function=self.evaluation_function
		)
		if self.contains_task(task_id=task.id):
			if ignore_error:
				return None
			else:
				raise RuntimeError(f'task {task} already exists in project {self.name}')
		return task

	def produce_tasks(self, ignore_error=False, echo=True):
		task_count = 0
		for data_id in self._data_id_prefixes:
			for estimator_name_and_id, estimator_class_and_arguments in self._estimators.items():
				estimator_name, estimator_id = estimator_name_and_id
				task = self.produce_task(
					estimator_name=estimator_name, estimator_id=estimator_id,
					estimator_class=estimator_class_and_arguments['class'],
					estimator_arguments=estimator_class_and_arguments['arguments'],
					data_id_prefix=data_id,
					ignore_error=ignore_error
				)
				if task is not None:
					task_count += 1
					self._pre_to_do[task.id] = task
		if echo:
			print(f'{task_count} tasks produced for project {self.name}')

	def fill_to_do_list(self, num_tasks=1, method='upper_bound', random_state=None, echo=True):
		if num_tasks > self.num_new_tasks:
			raise ValueError(f'num_tasks {num_tasks} is too large! There are only {self.num_new_tasks} available')
		data = DataFrame.from_records([
			{
				'estimator_name': task.estimator_name, 'estimator_id': task.estimator_id, 'data_id': task.data_id_prefix,
				'task_id': task.id
			}
			for task in self._pre_to_do.values()
		])
		if method == 'upper_bound':
			mean_per_data = self.scoreboard.mean_score_per_data.rename(columns={'score': 'data_score'})
			data = data.merge(mean_per_data, on='data_id', how='left')

			best_per_estimator = self.scoreboard.best_possible_score_per_estimator
			best_per_estimator = best_per_estimator.rename(columns={'score': 'estimator_score'})
			data = data.merge(best_per_estimator, on=['estimator_name', 'estimator_id'], how='left')
			if random_state is not None:
				random.seed(random_state)
			data['random'] = random.uniform(size=data.shape[0])
			if data['estimator_score'].isnull().values.any():
				raise RuntimeError('there are nulls among estimator_scores')

			if self.scoreboard.lowest_is_best:
				ascending = [True, False, True]
			else:
				ascending = [False, True, False]

			data.sort_values(
				by=['estimator_score', 'data_score', 'random'],
				ascending=ascending,
				na_position='first',
				inplace=True
			)

		elif method == 'random':
			data = data.sample(frac=1)

		else:
			raise ValueError(f'method {method} is unknown!')

		filled_count = 0
		for index, row in data.head(num_tasks).iterrows():
			self._take_from_pre_and_add_to_to_do(task_id=row['task_id'])
			filled_count += 1

		if echo:
			print(f'{filled_count} to-do tasks added to project {self.name}')

	def process(self, task):
		"""
		:type task: LearningTask
		"""
		self._scoreboard.add_task_score(task=task)
