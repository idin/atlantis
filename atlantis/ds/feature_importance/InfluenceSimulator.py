import pandas as pd
import numpy as np
from ravenclaw import get_numeric_columns
from joblib import Parallel, delayed
from chronometry.progress import iterate, ProgressBar
from copy import deepcopy


def get_chunk_indices(num_indices, chunk_size):
	chunk_size = max(1, chunk_size)
	return [(i, min(i+chunk_size, num_indices)) for i in range(0, num_indices+1, chunk_size)]


def q1(x):
	return np.nanquantile(x, 0.25)


def q3(x):
	return np.nanquantile(x, 0.75)


class InfluenceSimulator:
	def __init__(self, data, function, num_threads=1, num_points=200):
		"""
		:type data: pd.DataFrame
		:type function: callable
		:type num_threads: int
		:type num_points: int
		"""
		self._data = data
		self._numeric_column_names = get_numeric_columns(data=data)
		self._column_names = list(data.columns)
		self._num_threads = num_threads
		self._means = None
		self._sds = None
		self._function = function
		self._num_points = num_points + 1 if num_points % 2 == 1 else num_points
		self._simulation_points = None
		self._influencers = {}
		# self._memory = {}
		self._original_output = self._function(data)
		self._calculate_statistics()

	def get_influencers(self, column_name):



		upper = self._sds[column_name]

		if upper == 0:
			if self._means[column_name] == 0:
				lower, upper = -1, 1
			else:
				lower, upper = -0.1 * self._means[column_name], 0.1 * self._means[column_name]
		else:
			lower = -upper
		step_size = (upper - lower) / self._num_points
		return list(np.arange(lower, 0, step_size)) + list(np.arange(0, upper + step_size, step_size))

	@property
	def parallel_process(self):
		return Parallel(n_jobs=self._num_threads, backend='threading', require='sharedmem')

	@property
	def numeric_columns(self):
		"""
		:rtype: list[pd.Series]
		"""
		return [self._data[column] for column in self._numeric_column_names]

	@property
	def columns(self):
		"""
		:rtype: list[pd.Series]
		"""
		return [self._data[column] for column in self._column_names]

	def _calculate_statistics(self):
		output = self._function(self._data)
		try:
			self._output_mean = np.nanmean(output)
			self._output_sd = np.nanstd(output)

			if self._num_threads == 1:

				self._means = {
					name: np.nanmean(column) for name, column in zip(self._numeric_column_names, self.numeric_columns)
				}
				self._sds = {
					name: np.nanstd(column) for name, column in zip(self._numeric_column_names, self.numeric_columns)
				}
				uniques = {
					name: np.unique(column) for name, column in zip(self._column_names, self.columns)
				}

				self._influencers = {
					column_name: self.get_influencers(column_name=column_name)
					for column_name in self._numeric_column_names
					if len(uniques[column_name]) > 10
				}
				self._uniques = {name: uniques[name] for name in uniques.keys() if name not in self._influencers}

			else:
				means = self.parallel_process(
					delayed(np.nanmean)(column) for column in self.numeric_columns
				)
				self._means = {name: mean for name, mean in zip(self._numeric_column_names, means)}

				sds = self.parallel_process(
					delayed(np.nanstd)(column) for column in self.numeric_columns
				)
				self._sds = {name: sd for name, sd in zip(self._numeric_column_names, sds)}

				uniques = self.parallel_process(
					delayed(np.unique)(column) for column in self.columns
				)
				_uniques = {name: unique_values for name, unique_values in zip(self._column_names, uniques)}

				influence_lists = self.parallel_process(
					delayed(self.get_influencers)(column_name) for column_name in self._numeric_column_names
					if len(_uniques[column_name]) > 10
				)
				self._influencers = {
					column_name: influence_list
					for column_name, influence_list in zip(self._numeric_column_names, influence_lists)
				}
				self._uniques = {name: _uniques[name] for name in _uniques.keys() if name not in self._influencers}

		except AttributeError:
			self._output_mean = None
			self._output_sd = None

	def get_single_influence(self, column, perturbation, exact_value, function):
		new_data = self._data.drop(columns=column)

		if perturbation is not None and exact_value is None:
			new_data[column] = self._data[column] + perturbation
		elif perturbation is None and exact_value is not None:
			new_data[column] = exact_value
		else:
			raise ValueError(f'exactly one of perturbation and exact_value should be None. column: {column}')

		new_data = new_data[self._data.columns]
		output = function(new_data)
		difference = output - self._original_output
		return {
			'value': exact_value, 'perturbation': perturbation,
			'y_mean': np.nanmean(output), 'y_sd': np.nanstd(output),
			'y_q1': q1(output), 'y_median': np.median(output), 'y_q3': q3(output),
			'influence_mean': np.nanmean(difference), 'influence_sd': np.nanstd(difference),
			'influence_q1': q1(difference), 'influence_median': np.median(difference), 'influence_q3': q3(difference)
		}

	def simulate(self, echo=1, simulation_name='measuring influence'):
		"""
		:type echo: int
		:rtype: list[dict]
		"""
		def _get_influence(column, perturbation, exact_value, function):
			influence_result = self.get_single_influence(
				column=column,
				perturbation=perturbation,
				exact_value=exact_value,
				function=function
			)
			influence_result['column'] = column
			influence_result['perturbation'] = perturbation
			return influence_result

		progress_bar = ProgressBar(total=2 + len(self._influencers), echo=echo)

		if self._num_threads == 1:
			result = [
				_get_influence(column=column, perturbation=perturbation, exact_value=None, function=self._function)
				for column, influences in iterate(
					self._influencers.items(),
					progress_bar=progress_bar,
					text=f'{simulation_name} ...'
				)
				for perturbation in influences
			] + [
				_get_influence(column=column, perturbation=None, exact_value=unique_value, function=self._function)
				for column, unique_values in iterate(
					self._uniques.items(),
					progress_bar=progress_bar,
					text=f'{simulation_name} ...'.replace('influence', 'values')
				)
				for unique_value in unique_values
			]
		else:
			result = list(self.parallel_process(
				delayed(_get_influence)(
					column=column, perturbation=perturbation, exact_value=None, function=deepcopy(self._function)
				)
				for column, influences in iterate(
					self._influencers.items(),
					progress_bar=progress_bar,
					text=f'{simulation_name} in parallel ...'
				)
				for perturbation in influences
			)) + list(self.parallel_process(
				delayed(_get_influence)(
					column=column, perturbation=None, exact_value=unique_value, function=deepcopy(self._function)
				)
				for column, unique_values in iterate(
					self._uniques.items(),
					progress_bar=progress_bar,
					text=f'{simulation_name} in parallel ...'.replace('influence', 'values')
				)
				for unique_value in unique_values
			))

		progress_bar._total = 2 + len(self._influencers)
		progress_bar.show(amount=1 + len(self._influencers))

		result_data = pd.DataFrame.from_records(result)
		progress_bar.show(amount=2 + len(self._influencers))
		return result_data
