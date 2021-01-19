from pandas import DataFrame, concat
import numpy as np

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
		self._data_ids = set()

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
		self._estimator_ranks = None
		self._data_ranks = None

	@property
	def lowest_is_best(self):
		return self._lowest_is_best

	def _add_estimator_data_combination(self, estimator_name, estimator_id, data_id):
		key = estimator_name, estimator_id, data_id
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
		for data_id in self.data_ids:
			self._add_estimator_data_combination(
				estimator_name=estimator_name, estimator_id=estimator_id, data_id=data_id
			)

	def add_data_id(self, data_id):
		if data_id in self._data_ids:
			raise KeyError(f'data_id: {data_id} already exists!')
		self._data_ids.add(data_id)
		for estimator_name, estimator_id in self.estimators:
			self._add_estimator_data_combination(
				estimator_name=estimator_name, estimator_id=estimator_id, data_id=data_id
			)

	@property
	def data_ids(self):
		"""
		:rtype: set
		"""
		return self._data_ids

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

	def add_score(self, estimator_name, estimator_id, data_id, score_dictionary):
		if not isinstance(score_dictionary, dict):
			raise TypeError(f'score_dictionary should be a dict but it is of type {type(score_dictionary)}')
		try:
			score = self._unmeasured[(estimator_name, estimator_id, data_id)]
		except KeyError as e:
			display(self._unmeasured)
			raise e
		score.score_dictionary = score_dictionary

		self._measured[(estimator_name, estimator_id, data_id)] = score
		self.make_stale()

		del self._unmeasured[(estimator_name, estimator_id, data_id)]

	def add_task_score(self, task):
		"""
		:type task: TrainingTestTask
		"""
		if task.status == 'done':
			self.add_score(
				estimator_name=task.estimator_name,
				estimator_id=task.estimator_id,
				data_id=task.data_id_prefix,
				score_dictionary=task.evaluation
			)
		else:
			raise RuntimeError(f'{task} is not done, it is {task.status}')

	def _get_measured_records(self):
		records = []
		for key, score in self._measured.items():
			estimator_name, estimator_id, data_id = key
			records.append({
				'estimator_name': estimator_name, 'estimator_id': estimator_id, 'data_id': data_id,
				'score': score.score
			})
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

	def _get_all_records_fill_unmeasured_with_best(self):
		records = []
		for key, score in self._measured.items():
			estimator_name, estimator_id, data_id = key
			records.append({
				'estimator_name': estimator_name, 'estimator_id': estimator_id, 'data_id': data_id,
				'score': score.score
			})
		for key in self._unmeasured.keys():
			estimator_name, estimator_id, data_id = key
			records.append({
				'estimator_name': estimator_name, 'estimator_id': estimator_id, 'data_id': data_id,
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
			if len(records) == 0:
				aggregate = DataFrame({'data_id': list(self.data_ids)})
				aggregate['score'] = None
			else:
				data = DataFrame.from_records(self._get_measured_records())
				data = data[['data_id', 'score']]
				aggregate = data.groupby('data_id').mean().reset_index()

			aggregate.sort_values(by='score', ascending=self.lowest_is_best, inplace=True)

			self._mean_score_per_data = aggregate
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
			data = data[['estimator_name', 'estimator_id', 'score']]
			aggregate = data.groupby(['estimator_name', 'estimator_id']).mean().reset_index()
			aggregate.sort_values(by='score', ascending=self.lowest_is_best, inplace=True)

			self._best_possible_score_per_estimator = aggregate
		return self._best_possible_score_per_estimator
