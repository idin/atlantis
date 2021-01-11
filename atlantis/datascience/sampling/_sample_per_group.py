
from pandas import concat, DataFrame

# shuffles data and adds a column with new index
def shuffle(data, col = None):
	result = data.sample(frac = 1).reset_index(drop = True)
	if col is not None:
		result[col] = result.index

	return result


# sample the data but make sure to have a balanced number of samples per each group
# the groups are defined by columns
def sample_per_group(data, group_by, ratio=None, n=None):
	"""

	:type data: DataFrame
	:type group_by: list of str
	:type ratio: float
	:type num_rows: int
	:return:
	"""

	# group the data
	data = data.copy()
	"""
	:type data: DataFrame
	"""
	data['__order1'] = data.index
	grouped = data.groupby(by = group_by)

	training_sets = []
	test_sets = []

	for name, group in grouped:
		if n is None:
			num_rows = round(group.shape[0] * ratio)
		else:
			num_rows = round(min(n, group.shape[0]))
		shuffled = shuffle(data=group).reset_index(drop=True)
		new_training_set = shuffled.iloc[:num_rows, ]
		if num_rows>group.shape[0]:
			# if all of the data ends up in training set, test set should be empty instead of an error
			new_test_set = shuffled[shuffled.index!=shuffled.index]
		else:
			new_test_set = shuffled.iloc[(num_rows + 1):, ]
		training_sets.append(new_training_set)
		test_sets.append(new_test_set)

	training = concat(training_sets)
	"""
	:type training: DataFrame
	"""
	test = concat(test_sets)
	"""
	:type test: DataFrame
	"""

	training.reset_index(drop=True, inplace=True)
	test.reset_index(drop=True, inplace=True)

	training.index = training['__order1'].values
	test.index = test['__order1'].values
	training.sort_index(inplace=True)
	test.sort_index(inplace=True)

	training.drop(axis=1, labels = '__order1', inplace=True)
	test.drop(axis=1, labels='__order1', inplace=True)


	return training, test


