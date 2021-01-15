from ..parallel_computing import TrainingTestTask
from pandas import DataFrame


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

	def _add_estimator_data_combination(self, estimator_type, estimator_id, data_id):
		key = estimator_type, estimator_id, data_id
		if key not in self._all_combinations:
			self._all_combinations.add(key)
			self._unmeasured[key] = Score(main_metric=self._main_metric)

	def add_estimator(self, estimator_type, estimator_id):
		key = estimator_type, estimator_id
		if key in self._estimators:
			raise KeyError(f'estimator {estimator_type}, {estimator_id} already exists!')
		self._estimators.add(key)
		for data_id in self.data_ids:
			self._add_estimator_data_combination(
				estimator_type=estimator_type, estimator_id=estimator_id, data_id=data_id
			)

	def add_data_id(self, data_id):
		if data_id in self._data_ids:
			raise KeyError(f'data_id: {data_id} already exists!')
		self._data_ids.add(data_id)
		for estimator_type, estimator_id in self.estimators:
			self._add_estimator_data_combination(
				estimator_type=estimator_type, estimator_id=estimator_id, data_id=data_id
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

	def add_score(self, estimator_type, estimator_id, data_id, score_dictionary):
		score = self._unmeasured[(estimator_type, estimator_id, data_id)]
		score.score_dictionary = score_dictionary

		self._measured[(estimator_type, estimator_id, data_id)] = score
		del self._unmeasured[(estimator_type, estimator_id, data_id)]

	def add_task_score(self, task):
		"""
		:type task: TrainingTestTask
		"""
		if task.status == 'done':
			self.add_score(
				estimator_type=task.estimator_type,
				estimator_id=task.estimator_id,
				data_id=task.data_id,
				score_dictionary=task.evaluation
			)
		else:
			raise RuntimeError(f'{task} is not done, it is {task.status}')

	@property
	def data(self):
		"""
		:rtype: DataFrame
		"""
		records = []
		for key, score in self._measured.items():
			estimator_name, estimator_id, data_id = key
			records.append({
				'estimator_name': estimator_name, 'estimator_id': estimator_id, 'data_id': data_id, 'score': score.score,
				'score_upper_bound': score.upper_bound,
				**score.score_dictionary
			})
		return DataFrame.from_records(records)

	def get_scores(self, method):
		records = []

		for key, score in self._measured.items():
			estimator_name, estimator_id, data_id = key
			records.append({
				'estimator_name': estimator_name, 'estimator_id': estimator_id, 'data_id': data_id,
				'score': score.score
			})

		if method == 'actual':
			return DataFrame.from_records(records)

		elif method in ['estimator_mean', 'mean']:
			data = DataFrame.from_records(records).rename(columns={'score': 'estimator_mean_score'})
			return data.drop(columns='data_id').groupby(['estimator_name', 'estimator_id']).mean().reset_index()

		elif method == 'data_mean':
			data = DataFrame.from_records(records).rename(columns={'score': 'data_mean_score'})
			return data.drop(columns=['estimator_name', 'estimator_id']).groupby('data_id').mean().reset_index()

		elif method in ['estimator_upper_bound', 'upper_bound']:
			for key, score in self._unmeasured.items():
				estimator_name, estimator_id, data_id = key
				records.append({
					'estimator_name': estimator_name, 'estimator_id': estimator_id, 'data_id': data_id,
					'score': self._best_score
				})

			data = DataFrame.from_records(records).rename(columns={'score': 'estimator_upper_bound_score'})
			return data.drop(columns='data_id').groupby(['estimator_name', 'estimator_id']).mean().reset_index()

		else:
			raise ValueError(f'method "{method}" is unknown!')

	@property
	def available_combinations(self):
		unmeasured_records = [
			{'estimator_name': estimator_name, 'estimator_id': estimator_id, 'data_id': data_id}
			for estimator_name, estimator_id, data_id in self._unmeasured.keys()
		]
		return DataFrame.from_records(unmeasured_records)

	def choose_combinations(self, method='upper_bound', num_combinations=1):
		available_combinations = self.available_combinations

		if num_combinations > available_combinations.shape[1]:
			raise RuntimeError(
				f'number of combinations is more than available combinations: {available_combinations.shape[1]}'
			)

		if method in ['upper_bound', 'mean']:
			estimator_aggregate = self.get_scores(method=method)
			if method == 'upper_bound':
				score_column = 'estimator_upper_bound_score'
			else:
				score_column = 'estimator_mean_score'

			estimator_aggregate.rename(columns={score_column: 'estimator_score'}, inplace=True)

			combination_upper_bounds = available_combinations.merge(
				right=estimator_aggregate, on=['estimator_name', 'estimator_id'], how='left'
			)

			if combination_upper_bounds.shape[1] > 4:
				raise RuntimeError(f'data has more than 4 columns: {combination_upper_bounds.columns}')
			combination_upper_bounds = combination_upper_bounds[
				['estimator_name', 'estimator_id', 'data_id', 'estimator_score']
			]

			data_means = self.get_scores(method='data_mean').rename(columns={'data_mean_score': 'data_score'})

			combinations = combination_upper_bounds.merge(
				right=data_means, on='data_id', how='left'
			)
			if combinations.shape[1] > 5:
				raise RuntimeError(f'data has more than 5 columns: {combinations.columns}')
			combinations = combinations[
				['estimator_name', 'estimator_id', 'data_id', 'estimator_score', 'data_score']
			]

			if self._lowest_is_best:
				ascending = [True, False]
			else:
				ascending = [False, True]

			combinations = combinations.sort_values(by=['estimator_score', 'data_score'], ascending=ascending)

		else:
			raise ValueError(f'method "{method}" is unknown!')

		return combinations.head(num_combinations)
