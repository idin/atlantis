from ...exceptions import MissingArgumentError, AmbiguousArgumentsError
from ._get_cross_validation import get_cross_validation
from ._TrainingTestContainer import TrainingTestContainer

def get_training_test(
		data, sort_columns=None, id_columns=None, test_ratio=None, test_count=None, random_state=None
):
	"""
	:type data: DataFrame
	:type sort_columns: list[str]
	:type id_columns: list[str]
	:type test_ratio: float
	:type test_count: int
	:type random_state: int
	:rtype: TrainingTestContainer
	"""
	if test_ratio is None and test_count is None:
		raise MissingArgumentError('One of test_ratio or test_count should be provided!')
	elif test_ratio is not None and test_count is not None:
		raise AmbiguousArgumentsError('Only one of test_ratio or test_count should be provided!')
	elif test_ratio is None:
		test_ratio = test_count / data.shape[0]

	validation = get_cross_validation(
		data=data, num_splits=0, sort_columns=sort_columns, id_columns=id_columns,
		holdout_ratio=test_ratio, random_state=random_state,
	)

	return TrainingTestContainer(
		data=data, training_indices=validation.validation_indices, test_indices=validation.holdout_indices
	)
