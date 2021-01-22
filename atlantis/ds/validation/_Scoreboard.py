from pandas import DataFrame, concat
import numpy as np


def get_display_function():
	try:
		from IPython.core.display import display
		return display
	except ImportError:
		return False


class Score:
	def __init__(self, main_metric):
		self._main_metric = main_metric
		self._score_dictionary = None
		self._score = None

	@property
	def score(self):
		return self._score

	@score.setter
	def score(self, score):
		if self._score is not None:
			raise RuntimeError('cannot overwrite score!')
		self._score = score

	@property
	def score_dictionary(self):
		return self._score_dictionary

	@score_dictionary.setter
	def score_dictionary(self, score_dictionary):
		if self._score_dictionary is not None:
			raise RuntimeError('cannot overwrite score dictionary!')
		self._score_dictionary = score_dictionary
		self._score = score_dictionary[self._main_metric]


class Scoreboard:
	def __init__(self, main_metric=None, lowest_is_best=True, best_score=0):
		self._estimators = set()
		self._training_test_ids = set()

		if not lowest_is_best and best_score == 0:
			raise ValueError(f'best score of 0 does not work when the highest score is the best!')

		self._best_score = best_score

		self._all_combinations = set()
		self._measured = {}
		self._unmeasured = {}

		self._lowest_is_best = lowest_is_best
		self._main_metric = main_metric

		self._measured_data = None
		self._mean_score_per_data = None
		self._mean_score_per_estimator = None
		self._best_possible_score_per_estimator = None

	@property
	def lowest_is_best(self):
		return self._lowest_is_best

	def _add_estimator_data_combination(self, estimator_name, estimator_id, training_test_id):
		key = estimator_name, estimator_id, training_test_id
		if key not in self._all_combinations:
			self._all_combinations.add(key)
			self._unmeasured[key] = Score(main_metric=self._main_metric)

	def add_estimator(self, estimator_name, estimator_id):
		if not isinstance(estimator_name, str):
			raise TypeError(f'estimator_name should be str but it is of type {type(estimator_name)}')

		key = estimator_name, estimator_id
		if key in self._estimators:
			raise KeyError(f'estimator {key} already exists!')
		self._estimators.add(key)
		for training_test_id in self.training_test_ids:
			self._add_estimator_data_combination(
				estimator_name=estimator_name, estimator_id=estimator_id, training_test_id=training_test_id
			)

	def add_training_test_id(self, training_test_id):
		if training_test_id in self._training_test_ids:
			raise KeyError(f'training_test_id: {training_test_id} already exists!')
		self._training_test_ids.add(training_test_id)
		for estimator_name, estimator_id in self.estimators:
			self._add_estimator_data_combination(
				estimator_name=estimator_name, estimator_id=estimator_id, training_test_id=training_test_id
			)

	@property
	def training_test_ids(self):
		"""
		:rtype: set
		"""
		return self._training_test_ids

	@property
	def estimators(self):
		"""
		:rtype: set[(str, int)]
		"""
		return self._estimators

	def make_stale(self):
		self._measured_data = None
		self._mean_score_per_data = None
		self._mean_score_per_estimator = None
		self._best_possible_score_per_estimator = None

	def add_score(self, estimator_name, estimator_id, training_test_id, score_dictionary):
		if not isinstance(score_dictionary, dict):
			raise TypeError(f'score_dictionary should be a dict but it is of type {type(score_dictionary)}')
		try:
			score = self._unmeasured[(estimator_name, estimator_id, training_test_id)]
		except KeyError as e:
			display(self._unmeasured)
			raise e
		score.score_dictionary = score_dictionary

		self._measured[(estimator_name, estimator_id, training_test_id)] = score
		self.make_stale()

		del self._unmeasured[(estimator_name, estimator_id, training_test_id)]

	def add_task_score(self, task):
		"""
		:type task: LearningTask
		"""
		if task.status == 'done':
			self.add_score(
				estimator_name=task.estimator_name,
				estimator_id=task.estimator_id,
				training_test_id=task.training_test_id,
				score_dictionary=task.evaluation
			)
		else:
			raise RuntimeError(f'{task} is not done, it is {task.status}')

	def _get_measured_records(self, evaluation=False):
		if len(self.training_test_ids) == 0:
			raise RuntimeError('training_test_ids is empty')
		if len(self.estimators) == 0:
			raise RuntimeError('estimators is empty')

		records = []
		for key, score in self._measured.items():
			estimator_name, estimator_id, training_test_id = key
			record = {
				'estimator_name': estimator_name, 'estimator_id': estimator_id,
				'training_test_id': training_test_id,
				'score': score.score
			}
			if evaluation:
				record = {**record, **score.score_dictionary}
			records.append(record)
		return records

	@property
	def measured_data(self):
		"""
		:rtype: DataFrame
		"""

		if self._measured_data is None:
			data = DataFrame.from_records(self._get_measured_records())
			aggregate = data.groupby(['estimator_name', 'estimator_id']).agg(['count', 'mean', 'min', 'max', 'std'])
			aggregate.sort_values(by=('score', 'mean'), ascending=self.lowest_is_best, inplace=True)
			self._measured_data = aggregate
		return self._measured_data

	@property
	def evaluation_mean(self):
		records = self._get_measured_records(evaluation=True)
		data = DataFrame.from_records(records).groupby(['estimator_name', 'estimator_id']).mean().reset_index()
		return data

	def _get_all_records_fill_unmeasured_with_best(self):
		records = self._get_measured_records()
		for key in self._unmeasured.keys():
			estimator_name, estimator_id, training_test_id = key
			records.append({
				'estimator_name': estimator_name, 'estimator_id':
					estimator_id, 'training_test_id': training_test_id,
				'score': self._best_score
			})

		return records

	@property
	def mean_score_per_data(self):
		"""
		:rtype: DataFrame
		"""
		if self._mean_score_per_data is None:
			records = self._get_measured_records()
			all_data_sets = DataFrame({'training_test_id': list(self.training_test_ids)})
			if len(records) == 0:
				all_data_sets['score'] = None

			else:
				data = DataFrame.from_records(self._get_measured_records())
				data = data[['training_test_id', 'score']]
				aggregate = data.groupby('training_test_id').mean().reset_index()

				all_data_sets = all_data_sets.merge(aggregate, on='training_test_id', how='left')

			all_data_sets.sort_values(by='score', ascending=self.lowest_is_best, inplace=True, na_position='first')

			self._mean_score_per_data = all_data_sets
		return self._mean_score_per_data

	@property
	def mean_score_per_estimator(self):
		"""
		:rtype: DataFrame
		"""
		if self._mean_score_per_estimator is None:
			data = DataFrame.from_records(self._get_measured_records())
			data = data[['estimator_name', 'estimator_id', 'score']]
			self._mean_score_per_estimator = data.groupby(['estimator_name', 'estimator_id']).mean().reset_index()
			self._mean_score_per_estimator.sort_values(by='score', ascending=self.lowest_is_best, inplace=True)
		return self._mean_score_per_estimator

	@property
	def best_possible_score_per_estimator(self):
		"""
		:rtype: DataFrame
		"""
		if self._best_possible_score_per_estimator is None:
			data = DataFrame.from_records(self._get_all_records_fill_unmeasured_with_best())
			try:
				data = data[['estimator_name', 'estimator_id', 'score']]
			except:
				display(data)
				raise
			aggregate = data.groupby(['estimator_name', 'estimator_id']).mean().reset_index()
			aggregate.sort_values(by='score', ascending=self.lowest_is_best, inplace=True)

			self._best_possible_score_per_estimator = aggregate
		return self._best_possible_score_per_estimator

	def _repr_pretty_(self, p, cycle):
		if cycle:
			p.text(repr(self))
		else:
			self.display(p=p)

	def display(self, num_rows=None, p=None):
		display = get_display_function()
		measured_data = self.measured_data
		if num_rows is not None:
			measured_data = measured_data.head(num_rows)

		if display is False:
			print('scoreboard:')
			print(measured_data)
		else:
			print('scoreboard:')
			display(measured_data)