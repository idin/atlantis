from pandas import DataFrame
import random


def split_by_id(data, id_column, n=None, fraction=None, random_state=None, **kwargs):
	"""
	this method splits the data, by considering rows with the same id as one and refusing to divide them
	it is useful when you want a sample of the data based on ids when the ids are repeated
	and you want to take all or none of each id, but not a partial sample of multiple rows with the same id
	:param DataFrame data: dataframe to be sampled
	:param str id_column: column to be used as unique identifier
	:param int n: number of items to return
	:param float fraction: fraction of items to return, cannot be used with `n`
	:param bool replace: with or without replacement
	:param int random_state: seed for the random number generator
	:rtype: tuple[DataFrame, DataFrame]
	"""
	# find unique ids:
	ids = list(set(data[id_column]))

	# shuffle the ids:
	random.Random(random_state).shuffle(ids)

	# choose training and test ids
	if n is None:
		n = max(0, min(len(ids), round(fraction * len(ids))))
	training_ids = ids[:n]
	test_ids = ids[n:]

	data = data.copy()
	data['__index__'] = data.index
	data['__order__'] = range(len(data))

	# create training and test dataframes
	training_data = DataFrame({id_column: training_ids})
	test_data = DataFrame({id_column: test_ids})

	# merge with the rest of the data
	training_data = training_data.merge(
		right=data, on=id_column, how='left'
	).set_index('__index__').sort_values('__order__').drop(columns='__order__')
	training_data.index.name = data.index.name

	test_data = test_data.merge(
		right=data, on=id_column, how='left'
	).set_index('__index__').sort_values('__order__').drop(columns='__order__')
	test_data.index.name = data.index.name

	return training_data, test_data


def sample_by_id(data, id_column, n=None, fraction=None, replace=False, random_state=None, **kwargs):
	"""
	this method samples the data, by considering rows with the same id as one and refusing to divide them
	it is useful when you want a sample of the data based on ids when the ids are repeated
	and you want to take all or none of each id, but not a partial sample of multiple rows with the same id
	:param DataFrame data: dataframe to be sampled
	:param str id_column: column to be used as unique identifier
	:param int n: number of items to return
	:param float fraction: fraction of items to return, cannot be used with `n`
	:param bool replace: with or without replacement
	:param int random_state: seed for the random number generator
	:rtype: DataFrame
	"""
	data = data.copy()
	data['__index__'] = data.index
	ids = data[[id_column]].drop_duplicates()
	sampled_ids = ids.sample(n=n, frac=fraction, replace=replace, random_state=random_state, **kwargs)
	result = sampled_ids.merge(right=data, on=id_column, how='left').set_index('__index__')
	result.index.name = data.index.name
	return result
