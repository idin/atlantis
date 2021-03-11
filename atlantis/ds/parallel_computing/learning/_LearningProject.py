from pandas import concat, DataFrame
from numpy import random

from .._Project import Project
from ...evaluation import evaluate_regression, evaluate_classification
from .._DataSlice import TrainingTestSlice
from ...validation import Scoreboard, TrainingTestContainer
from ._LearningTask import LearningTask
from ....collections.OrderedSet import OrderedSet


class LearningProject(Project):
	def __init__(
			self, name, y_column, problem_type, x_columns=None,
			time_unit='ms', evaluation_function=None, main_metric=None, lowest_is_best=None, best_score=None,
			scoreboard=None, processor=None
	):
		"""

		:type 	name: str
		:type 	y_column: str

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
		self._estimators = {}
		self._training_test_slice_ids = OrderedSet()

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
				raise ValueError(
					'main_metric, lowest_is_best, and best_score should be provided for evaluation_function')

		if y_column is None:
			raise ValueError('y_column should be provided')
		self._y_column = y_column
		self._x_columns = x_columns

		if scoreboard is None:
			scoreboard = Scoreboard(main_metric=main_metric, lowest_is_best=lowest_is_best, best_score=best_score)
		self._scoreboard = scoreboard
		self._all_tasks_produced = False

	def __repr__(self):
		lines = [
			super().__repr__(),
			f'estimators: {len(self.estimators)}',
			f'training_test_slice_id: {self._training_test_slice_ids}',
		]
		return '\n'.join(lines)

	def _generate_training_test_slice_id(self):
		return f'{self.name}_{len(self._training_test_slice_ids) + 1}'

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

		if self._x_columns is None:
			self._x_columns = x_columns
		else:
			if set(self._x_columns) != set(x_columns):
				raise ValueError(f'x_columns cannot change!')

	def add_x_columns_from_data(self, data):
		if self.x_columns is None:
			self.x_columns = [column for column in data.column if column != self.y_column]

	def add_training_test_slice(
			self, training_test_slice, data=None, training_test_slice_id=None, overwrite=False,
			add_to_training_test_ids=True
	):
		"""
		:type training_test_slice_id: str
		:type training_test_slice: TrainingTestSlice
		:return:
		"""
		if training_test_slice_id is None:
			training_test_slice_id = self._generate_training_test_slice_id()

		if data is not None:
			self.processor.add_data(data_id=training_test_slice.data_id, data=data, overwrite=overwrite)

		columns = self.processor.get_obj(obj_type='columns', obj_id=training_test_slice.data_id)
		x_columns = [column for column in columns if column != self.y_column]
		self.x_columns = x_columns

		if add_to_training_test_ids:
			self._training_test_slice_ids.add(training_test_slice_id)

		self.processor.add_obj(
			obj_type='tts', obj_id=training_test_slice_id, obj=training_test_slice,
			overwrite=overwrite
		)

		if add_to_training_test_ids:
			self.scoreboard.add_training_test_id(training_test_id=training_test_slice_id)
			self._all_tasks_produced = False

	def add_training_test_container(self, container, training_test_slice_id=None, overwrite=False):
		"""
		:type training_test_slice_id: str
		:type container: TrainingTestContainer
		:type overwrite: bool
		"""
		if training_test_slice_id is None:
			training_test_slice_id = self._generate_training_test_slice_id()

		self.add_training_test_slice(
			training_test_slice_id=training_test_slice_id,
			data=container.data,
			overwrite=overwrite,
			training_test_slice=TrainingTestSlice(
				data_id=training_test_slice_id,
				training_indices=container.training_indices,
				test_indices=container.test_indices
			)
		)

	def add_training_test_data(self, training_data, test_data, training_test_slice_id=None, overwrite=False):
		if training_test_slice_id is None:
			training_test_slice_id = self._generate_training_test_slice_id()

		data = concat([training_data, test_data]).reset_index(drop=True)
		training_indices = list(range(training_data.shape[0]))
		test_indices = [x + training_data.shape[0] for x in range(test_data.shape[0])]
		tts = TrainingTestSlice(
			data_id=training_test_slice_id,
			training_indices=training_indices,
			test_indices=test_indices
		)
		self.add_training_test_slice(
			training_test_slice=tts, data=data, training_test_slice_id=training_test_slice_id,
			overwrite=overwrite
		)

	@staticmethod
	def _get_estimator_name(estimator_class):
		return estimator_class.__name__

	def add_estimator(self, estimator_class, estimator_id, estimator_arguments):
		if not isinstance(estimator_class, type):
			raise TypeError(f'estimator_class is of type {type(estimator_class)}')

		estimator_name = self._get_estimator_name(estimator_class)
		key = estimator_name, estimator_id
		self._estimators[key] = {'class': estimator_class, 'arguments': estimator_arguments}
		self.scoreboard.add_estimator(estimator_name=estimator_name, estimator_id=estimator_id)
		self._all_tasks_produced = False
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
				estimator_class=estimator,
				estimator_id=estimator_id,
				estimator_arguments=estimator_arguments
			)

	@property
	def estimators(self):
		"""
		:rtype: dict[(str, int), dict]
		"""
		return self._estimators

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
	def scoreboard(self):
		"""
		:rtype: Scoreboard
		"""
		return self._scoreboard

	def produce_task(
			self, estimator_name, estimator_id, estimator_class, estimator_arguments,
			training_test_slice_id,
			ignore_error=False
	):
		"""

		:param estimator_name:
		:param estimator_id:
		:param estimator_class:
		:param estimator_arguments:
		:param training_test_slice_id:
		:param ignore_error:
		:rtype: LearningTask
		"""
		if self.x_columns is None:
			raise RuntimeError(f'x_columns is None')

		task = LearningTask(
			project_name=self.name, estimator_class=estimator_class,
			estimator_name=estimator_name, estimator_id=estimator_id,
			estimator_arguments=estimator_arguments,
			training_test_slice_id=training_test_slice_id,
			y_column=self.y_column, x_columns=self.x_columns,
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
		for training_test_slice_id in self._training_test_slice_ids:
			for estimator_name_and_id, estimator_class_and_arguments in self._estimators.items():
				estimator_name, estimator_id = estimator_name_and_id
				task = self.produce_task(
					estimator_name=estimator_name, estimator_id=estimator_id,
					estimator_class=estimator_class_and_arguments['class'],
					estimator_arguments=estimator_class_and_arguments['arguments'],
					training_test_slice_id=training_test_slice_id,
					ignore_error=ignore_error
				)
				if task is not None:
					task_count += 1

					self._pre_to_do[task.id] = task

		if echo:
			print(f'{task_count} tasks produced for project {self.name}')
		self._all_tasks_produced = True

	def _get_new_to_do_list(self, num_tasks=1, method='upper_bound', random_state=None, echo=True):
		if num_tasks > self.new_count:
			raise ValueError(f'num_tasks {num_tasks} is too large! There are only {self.new_count} available')
		if len(self._pre_to_do) == 0:
			if echo:
				print('no new tasks to fill the to-do list')
			return None

		data = DataFrame.from_records([
			{
				'estimator_name': task.estimator_name, 'estimator_id': task.estimator_id,
				'training_test_id': task.training_test_id,
				'task_id': task.id
			}
			for task in self._pre_to_do.values()
		])
		if method == 'upper_bound':
			mean_per_data = self.scoreboard.mean_score_per_data.rename(columns={'score': 'data_score'})
			data = data.merge(mean_per_data, on='training_test_id', how='left')

			best_per_estimator = self.scoreboard.best_possible_score_per_estimator
			best_per_estimator = best_per_estimator.rename(columns={'score': 'estimator_score'})
			data = data.merge(best_per_estimator, on=['estimator_name', 'estimator_id'], how='left')
			if random_state is not None:
				random.seed(random_state)

			# we want to try the estimators that have the best shot
			# also we want to avoid repeating the same estimator
			# (hence estimator_repetition counts the number of
			# times an estimator appears, we put estimator_repetition first
			# because it is useless if it comes after estimator_score
			# and we want to prioritize data sets that have the worst outcome, sooner rather than later
			# on top of all that, when everything is equal, randomize

			data['random'] = random.uniform(size=data.shape[0])
			if data['estimator_score'].isnull().values.any():
				raise RuntimeError('there are nulls among estimator_scores')

			ascending = [self.scoreboard.lowest_is_best, not self.scoreboard.lowest_is_best, True]

			data.sort_values(
				by=['estimator_score', 'data_score', 'random'],
				ascending=ascending,
				na_position='first',
				inplace=True
			)

			data['estimator_repetition'] = data.groupby(['estimator_name', 'estimator_id']).cumcount()
			data.sort_values(
				by=['estimator_repetition', 'estimator_score', 'data_score', 'random'],
				ascending=[True] + ascending,
				na_position='first',
				inplace=True
			)

		elif method == 'random':
			data = data.sample(frac=1, random_state=random_state)

		else:
			raise ValueError(f'method {method} is unknown!')

		return [row['task_id'] for index, row in data.head(num_tasks).iterrows()]

	def fill_to_do_list(self, num_tasks=1, method='upper_bound', random_state=None, echo=True):
		task_ids = self._get_new_to_do_list(num_tasks=num_tasks, method=method, random_state=random_state, echo=echo)

		filled_count = 0
		for task_id in task_ids:
			self._take_from_pre_and_add_to_to_do(task_id=task_id)
			filled_count += 1

		if echo:
			print(f'{filled_count} to-do tasks added to project {self.name}')

	def process(self, task):
		"""
		:type task: LearningTask
		"""
		if task.has_error():
			for error, trace in task.errors:
				print(f'trace: {trace}')
			raise task.errors[0][0]
		if task.status != 'done':
			raise RuntimeError(f'task is not done. Task status is {task.status}')

		if not isinstance(task.evaluation, dict):
			raise TypeError(f'evaluation is of type {type(task.evaluation)}')
		self._scoreboard.add_task_score(task=task)

	def get_best_estimators(self, num_estimators=1):
		scores = self.scoreboard.mean_score_per_estimator.sort_values(
			'score', ascending=self.scoreboard.lowest_is_best
		)
		if scores.shape[0] < num_estimators:
			raise RuntimeError(f'there are only {scores.shape[0]} estimators to choose from!')
		result = []
		for row_num, row in scores.head(num_estimators).iterrows():
			estimator_name = row['estimator_name']
			estimator_id = row['estimator_id']
			score = row['score']
			dictionary = self.estimators[(estimator_name, estimator_id)]
			parameters = {
				'name': estimator_name, 'id': estimator_id, 'class': dictionary['class'], 
				'arguments': dictionary['arguments'], 'score': score
			}
			result.append(parameters)
		return result

	def get_best_estimator(self):
		return self.get_best_estimators(num_estimators=1)[0]
