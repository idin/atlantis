from pandas import concat, DataFrame
from ...exceptions import MissingArgumentError, AmbiguousArgumentsError
from ._split_data import split_data

from ._TrainingTestContainer import TrainingTestContainer
from ._ValidationContainer import ValidationContainer


def get_cross_validation(
		data, num_splits, holdout_ratio=None, holdout_count=None, sort_columns=None, id_columns=None,
		test_ratio=None, test_count=None, random_state=None,
		min_training_ratio=None, min_training_count=None
):
	"""
	:type 	data: DataFrame
	:type 	num_splits: int
	:type 	holdout_ratio: float
	:type 	holdout_count: int
	:type 	sort_columns: list[str]
	:type 	id_columns: list[str]
	:type 	test_ratio: float
	:type 	test_count: int
	:type 	random_state: int
	:type 	min_training_ratio: float
	:type 	min_training_count: int
	:rtype: ValidationContainer
	"""
	if id_columns is None and sort_columns is not None:
		id_columns = sort_columns
		time_series = True
	elif id_columns is not None and sort_columns is None:
		time_series = False
	elif id_columns is not None and sort_columns is not None:
		raise ValueError('Only one of sort_columns and id_columns can be None')
	else:
		time_series = False

	if test_ratio is None and test_count is None:
		test_ratio = holdout_ratio
		test_count = holdout_count
	elif test_ratio is not None and test_count is not None:
		raise AmbiguousArgumentsError('Only one of test_ratio and test_count should be provided!')

	if id_columns is None or id_columns == []:
		new_data = data.copy()
		id_columns = ['cross_validation_id']
		new_data['cross_validation_id'] = range(new_data.shape[0])
	else:
		if isinstance(id_columns, str):
			id_columns = [id_columns]
		new_data = data[id_columns].copy()

	new_data['cross_validation_index'] = range(new_data.shape[0])
	new_data['cross_validation_count'] = 1
	data_ids = new_data[id_columns + ['cross_validation_count']].groupby(id_columns).sum().reset_index(drop=False)
	total_count = data_ids['cross_validation_count'].sum()

	if holdout_ratio is not None and holdout_count is not None:
		raise AmbiguousArgumentsError('Only one of holdout_count and holdout_ratio should be provided!')
	elif holdout_ratio is not None:
		validation_holdout_counts = None
		validation_holdout_ratios = [1 - holdout_ratio, holdout_ratio]
	else:
		holdout_count = holdout_count or 0
		validation_holdout_counts = [total_count - holdout_count, holdout_count]
		validation_holdout_ratios = None

	# randomize for regular cross validation or sort for timeseries cross validation
	if time_series:
		data_ids = data_ids.sort_values(by=id_columns)
	else:
		data_ids = data_ids.sample(frac=1, random_state=random_state)

	validation_ids, holdout_ids = split_data(
		data=data_ids, count_column='cross_validation_count',
		counts=validation_holdout_counts, ratios=validation_holdout_ratios
	)

	def _get_specific_data(_id_data):
		if _id_data is None:
			return None
		return _id_data.merge(right=new_data, on=id_columns, how='left')  # .reset_index(drop=True)

	def _get_indices_from_specific_data(_specific_data):
		if _specific_data is None:
			return None
		return sorted(list(_specific_data['cross_validation_index']))

	validation_data = _get_specific_data(validation_ids)
	validation_indices = _get_indices_from_specific_data(validation_data)
	holdout_data = _get_specific_data(holdout_ids)
	holdout_indices = _get_indices_from_specific_data(holdout_data)

	if time_series:
		if num_splits + 1 > validation_ids.shape[0]:
			raise ValueError(f'maximum number of splits for this data is {validation_ids.shape[0] - 1}')
	else:
		if num_splits > validation_ids.shape[0]:
			raise ValueError(f'maximum number of splits for this data is {validation_ids.shape[0]}')

	if time_series:
		validation_ids = validation_ids.sort_values(by=id_columns)
	else:
		validation_ids = validation_ids.sample(frac=1, random_state=random_state)
	validation_ids = validation_ids  # .reset_index(drop=True)

	validation_total = validation_ids['cross_validation_count'].sum()

	if test_ratio is not None and test_count is not None:
		raise AmbiguousArgumentsError('Only one of test_count and test_ratio should be provided!')

	if num_splits == 0:
		fold_indices = []

	elif num_splits == 1:
		if test_ratio is not None:
			training_test_counts = None
			training_test_ratios = [1 - test_ratio, test_ratio]
		elif test_count is None:
			raise MissingArgumentError('Either test count or test ratio should be provided!')
		else:
			training_test_counts = [validation_total - test_count, test_count]
			training_test_ratios = None

		training_ids, test_ids = split_data(
			data=validation_ids, count_column='cross_validation_count',
			counts=training_test_counts, ratios=training_test_ratios
		)

		training_indices = _get_indices_from_specific_data(_get_specific_data(training_ids))
		test_indices = _get_indices_from_specific_data(_get_specific_data(test_ids))
		fold_indices = [{'training': training_indices, 'test': test_indices}]

	else:

		fold_indices = []
		if time_series:
			if min_training_count is None:
				if min_training_ratio is None:
					min_training_count = 0
				else:
					min_training_count = min_training_ratio * validation_ids.shape[0]

			first_count = max(min_training_count, validation_total / (num_splits + 1))
			other_counts = (validation_total - first_count) / num_splits
			counts = [first_count] + [other_counts] * num_splits
			splits = split_data(
				data=validation_ids, count_column='cross_validation_count',
				counts=counts, ratios=None
			)
			for i in range(num_splits):
				training_ids = concat(splits[:i + 1])
				test_ids = splits[i + 1]
				training_indices = _get_indices_from_specific_data(_get_specific_data(training_ids))
				test_indices = _get_indices_from_specific_data(_get_specific_data(test_ids))
				fold_indices.append({'training': training_indices, 'test': test_indices})

		else:

			counts = [validation_total / num_splits] * num_splits
			splits = split_data(
				data=validation_ids, count_column='cross_validation_count',
				counts=counts, ratios=None
			)
			for i in range(num_splits):
				training_ids = concat([splits[j] for j in range(num_splits) if j != i])
				test_ids = splits[i]
				training_indices = _get_indices_from_specific_data(_get_specific_data(training_ids))
				test_indices = _get_indices_from_specific_data(_get_specific_data(test_ids))
				fold_indices.append({'training': training_indices, 'test': test_indices})

	if fold_indices is not None:
		folds = [
			TrainingTestContainer(
				data=data, training_indices=fold['training'], test_indices=fold['test'],
				sort_columns=sort_columns
			)
			for fold in fold_indices
		]
	else:
		folds = []

	return ValidationContainer(
		data=data, validation_indices=validation_indices, holdout_indices=holdout_indices,
		folds=folds,
		sort_columns=sort_columns
	)



	return TrainingTestContainer(
		data=data, training_indices=validation.validation_indices, test_indices=validation.holdout_indices,
		sort_columns=sort_columns
	)
