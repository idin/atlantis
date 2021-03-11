from pandas import DataFrame
from ._get_cross_validation import get_cross_validation
from ._get_training_test import get_training_test
from ._ValidationContainer import ValidationContainer
from ._TrainingTestContainer import TrainingTestContainer
from ...exceptions import MissingArgumentError


class Validation:
	def __init__(
			self, num_splits, id_columns=None, sort_columns=None, random_state=None,
			holdout_ratio=None, holdout_count=None, test_ratio=None, test_count=None,
			min_training_count=None, min_training_ratio=None
	):
		"""
		:type id_columns: NoneType or list[str] or str
		:type sort_columns: NoneType or list[str] or str
		:type random_state: NoneType or int or float
		"""
		if isinstance(id_columns, str):
			id_columns = [id_columns]
		if isinstance(sort_columns, str):
			sort_columns = [sort_columns]

		self._num_splits = num_splits
		self._id_columns = id_columns
		self._sort_columns = sort_columns
		self._random_state = random_state

		self._holdout_ratio = holdout_ratio
		self._holdout_count = holdout_count
		self._test_ratio = test_ratio
		self._test_count = test_count
		self._min_training_count = min_training_count
		self._min_training_ratio = min_training_ratio

	def split(self, data, random_state=None):
		"""
		:type data: DataFrame
		:type random_state: int
		:rtype: ValidationContainer
		"""
		return get_cross_validation(
			data=data,
			num_splits=self._num_splits,
			holdout_ratio=self._holdout_ratio,
			holdout_count=self._holdout_count,
			test_count=self._test_count,
			sort_columns=self._sort_columns,
			id_columns=self._id_columns,
			test_ratio=self._test_ratio,
			random_state=random_state or self._random_state,
			min_training_count=self._min_training_count,
			min_training_ratio=self._min_training_ratio
		)


class TrainingTest(Validation):
	def __init__(self, id_columns=None, sort_columns=None, random_state=None, test_ratio=None, test_count=None):
		"""
		:param id_columns:
		:param sort_columns:
		:param random_state:
		:param test_ratio:
		:param test_count:
		"""
		super().__init__(
			num_splits=None, id_columns=id_columns, sort_columns=sort_columns, random_state=random_state,
			holdout_ratio=None, holdout_count=None, test_count=test_count, test_ratio=test_ratio,
			min_training_count=None, min_training_ratio=None
		)

	def split(self, data, random_state=None):
		"""
		:type data: DataFrame
		:type random_state: int
		:rtype: TrainingTestContainer
		"""
		return get_training_test(
			data=data,
			sort_columns=self._sort_columns,
			id_columns=self._id_columns,
			test_ratio=self._test_ratio,
			test_count=self._test_count,
			random_state=random_state or self._random_state
		)


class TimeSeriesTrainingTest(TrainingTest):
	def __init__(self, sort_columns, random_state=None, test_ratio=None, test_count=None):
		"""
		:param id_columns:
		:param sort_columns:
		:param random_state:
		:param test_ratio:
		:param test_count:
		"""
		super().__init__(
			id_columns=None, sort_columns=sort_columns, random_state=random_state,
			test_ratio=test_ratio, test_count=test_count
		)


class CrossValidation(Validation):
	def __init__(
			self, num_splits, id_columns=None, random_state=None,
			holdout_ratio=None, holdout_count=None, test_ratio=None, test_count=None,
			min_training_count=None, min_training_ratio=None
	):
		"""
		:param num_splits:
		:param id_columns:
		:param random_state:
		:param holdout_ratio:
		:param holdout_count:
		:param test_ratio:
		:param test_count:
		:param min_training_count:
		:param min_training_ratio:
		"""
		super().__init__(
			num_splits=num_splits,
			id_columns=id_columns, sort_columns=None, random_state=random_state,
			holdout_ratio=holdout_ratio, holdout_count=holdout_count, test_ratio=test_ratio,
			test_count=test_count,
			min_training_count=min_training_count, min_training_ratio=min_training_ratio
		)


class TimeSeriesValidation(Validation):
	def __init__(
			self, num_splits, sort_columns, random_state=None,
			holdout_ratio=None, holdout_count=None, test_ratio=None, test_count=None,
			min_training_count=None, min_training_ratio=None
	):
		"""
		:param num_splits:
		:param sort_columns:
		:param random_state:
		:param holdout_ratio:
		:param holdout_count:
		:param test_ratio:
		:param test_count:
		:param min_training_count:
		:param min_training_ratio:
		"""
		if sort_columns is None:
			raise MissingArgumentError('sort_columns should be provided!')

		super().__init__(
			num_splits=num_splits,
			id_columns=None, random_state=random_state, sort_columns=sort_columns,
			holdout_ratio=holdout_ratio, holdout_count=holdout_count, test_ratio=test_ratio,
			test_count=test_count,
			min_training_count=min_training_count, min_training_ratio=min_training_ratio
		)
