from pandas import DataFrame

def join_and_keep_order(left, right, remove_duplicates, keep='first', **kwargs):
	"""
	:type left: DataFrame
	:type right: DataFrame
	:rtype: DataFrame
	"""
	left = left.copy()
	right = right.copy()
	left['_left_id'] = range(left.shape[0])
	right['_right_id'] = range(right.shape[0])
	result = left.merge(right=right, **kwargs)
	result.sort_values(axis='index', by=['_left_id', '_right_id'], inplace=True)
	if remove_duplicates: result = result[(~result['_left_id'].duplicated(keep=keep)) & (~result['_right_id'].duplicated(keep=keep))]
	return result.drop(columns=['_left_id', '_right_id'])


def join_wisely(left, right, remove_duplicates=True, echo=False, **kwargs):
	"""
	joins two dataframes and returns a dictionary with 3 members: left_only, right_only, and both (the results of the two joins)
	:type left: DataFrame
	:type right: DataFrame
	:type kwargs: dict
	:rtype: dict of DataFrames
	"""
	left=left.copy()
	right=right.copy()

	left['_left_id'] = range(left.shape[0])
	right['_right_id'] = range(right.shape[0])

	both_data = left.merge(right=right, how='inner', **kwargs)
	if remove_duplicates:
		both_data.sort_values(axis=0, by=['_left_id', '_right_id'], inplace=True)
		left_id_duplicated = both_data._left_id.duplicated(keep='first')
		right_id_duplicated = both_data._right_id.duplicated(keep='first')
		both_data = both_data[~left_id_duplicated & ~right_id_duplicated]
	left_only_data = left[~left['_left_id'].isin(both_data['_left_id'])].copy()
	right_only_data = right[~right['_right_id'].isin(both_data['_right_id'])].copy()

	both_data.drop(labels=['_left_id', '_right_id'], axis=1, inplace=True)
	left_only_data.drop(labels='_left_id', axis=1, inplace=True)
	right_only_data.drop(labels='_right_id', axis=1, inplace=True)
	if echo:
		print(f'left:{left.shape}, right:{right.shape}\nboth:{both_data.shape}, left_only:{left_only_data.shape}, right_only:{right_only_data.shape}')

	return {'both':both_data, 'left_only':left_only_data, 'right_only':right_only_data}

