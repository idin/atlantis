import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import warnings
from copy import deepcopy

from ...time.progress import ProgressBar, iterate

from .InfluenceSimulator import InfluenceSimulator


def q1(x):
	return np.nanquantile(x, 0.25)


def q3(x):
	return np.nanquantile(x, 0.75)


def get_single_column_influence(function, data, x_column, num_deltas):
	if num_deltas % 2 == 1:
		num_deltas += 1

	new_data = deepcopy(data)

	x_mean = new_data[x_column].mean()
	x_sd = new_data[x_column].std()
	x_lower = -x_sd
	x_upper = x_sd
	if x_upper == x_lower:
		if x_mean == 0:
			x_upper = 1
			x_lower = -1
		else:
			x_upper = x_mean * 0.1
			x_lower = -x_mean * 0.1
	step_size = (x_upper - x_lower) / num_deltas
	delta_data = pd.DataFrame({
		'x_delta': list(np.arange(x_lower, 0, step_size)) + list(np.arange(0, x_upper + step_size, step_size))
	})

	new_data['variable'] = x_column
	delta_data['variable'] = x_column
	delta_data['_row_'] = range(len(delta_data))
	new_data = new_data.merge(right=delta_data, on='variable', how='inner')

	test_data = new_data.drop(columns=['x_delta', 'variable', '_row_'])

	new_data['_original_y_'] = function(test_data)
	test_data[x_column] = test_data[x_column] + new_data['x_delta']
	new_data['_new_y_'] = function(test_data)
	new_data['y_delta'] = new_data['_new_y_'] - new_data['_original_y_']
	new_data['delta_ratio'] = new_data['y_delta'] / new_data['x_delta']
	new_data['_count_'] = 1

	aggregate = new_data[['variable', '_row_', '_new_y_', '_original_y_', '_count_', 'y_delta', 'delta_ratio']].groupby(
		['variable', '_row_']
	).agg(
		mean=('_new_y_', np.mean), median=('_new_y_', np.median),
		q1=('_new_y_', q1), q3=('_new_y_', q3),
		min=('_new_y_', np.min), max=('_new_y_', np.max),
		original_mean=('_original_y_', np.mean),
		count=('_count_', np.sum),
		mean_y_delta=('y_delta', np.mean),
		median_y_delta=('y_delta', np.median),
		mean_delta_ratio=('delta_ratio', np.mean),
		median_delta_ratio=('delta_ratio', np.median)
	).reset_index().merge(right=delta_data[['_row_', 'x_delta']], on='_row_', how='left').drop(columns=['_row_'])

	return aggregate


def get_function_influence(function, data, x_columns, num_deltas=200, num_threads=1, echo=1):
	"""
	:param callable function: function that gives a y for a given data
	:param pd.DataFrame data:
	:param list[str] or str x_columns:
	:param int num_deltas:
	:param int num_threads:
	:type echo: int
	:rtype: pd.DataFrame
	"""
	if isinstance(x_columns, str):
		x_columns = [x_columns]
	elif not isinstance(x_columns, list):
		raise TypeError(f'x_columns should be either a string or a list not a {type(x_columns)}')

	def get_influence_data(the_tuple):
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			x_column, this_data, this_num_deltas, this_function = the_tuple
			record = get_single_column_influence(
				function=this_function, data=this_data, x_column=x_column, num_deltas=this_num_deltas
			)
		return record

	tuples = [
		(x_column, data, num_deltas, function)
		for x_column in iterate(x_columns, echo=echo, text='preparing data ...')
	]

	progress_bar = ProgressBar(total=len(x_columns) + 1, echo=echo)
	progress_bar.show(amount=len(x_columns), text='data prepared.')

	if num_threads == 1:
		dataframes = [
			get_influence_data(t)
			for t in iterate(iterable=tuples, progress_bar=progress_bar, text='measuring influence ...')
		]
	else:
		dataframes = Parallel(n_jobs=num_threads, backend='threading', require='sharedmem')(
			delayed(get_influence_data)(t)
			for t in iterate(iterable=tuples, progress_bar=progress_bar, text='measuring influence ...')
		)

	result = pd.concat(dataframes).sort_values(by=['variable', 'x_delta'])
	progress_bar.show(amount=len(dataframes) + 1, text='influence measured.')
	return result


def get_model_influence(model, data, x_columns, use_probability=False, num_points=200, num_threads=1, echo=1):
	if use_probability:
		function = model.predict_proba
		simulation_name = 'measuring influence on probability'

	else:
		simulation_name = 'measuring influence'
		try:
			function = model.predict
		except AttributeError:
			function = model.pred

	simulator = InfluenceSimulator(
		data=data[x_columns], function=function, num_threads=num_threads, num_points=num_points
	)
	return simulator.simulate(echo=echo, simulation_name=simulation_name)
