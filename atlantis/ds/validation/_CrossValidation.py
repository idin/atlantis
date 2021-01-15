from pandas import DataFrame
from ._get_cross_validation import get_cross_validation, get_training_test
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
	def split(self, data, random_state=None):
		return get_training_test(
			data=data,
			sort_columns=self._sort_columns,
			id_columns=self._id_columns,
			test_ratio=self._test_ratio,
			test_count=self._test_count,
			random_state=random_state or self._random_state
		)


class CrossValidation(Validation):
	def __init__(
			self, id_columns=None, random_state=None,
			holdout_ratio=None, holdout_count=None, test_ratio=None, test_count=None,
			min_training_count=None, min_training_ratio=None
	):
		"""
		:type id_columns: NoneType or list[str] or str
		:type random_state: NoneType or int or float
		"""
		super().__init__(
			id_columns=id_columns, sort_columns=None, random_state=random_state,
			holdout_ratio=holdout_ratio, holdout_count=holdout_count, test_ratio=test_ratio, test_count=test_count,
			min_training_count=min_training_count, min_training_ratio=min_training_ratio
		)


class TimeSeriesValidation(Validation):
	def __init__(
			self, sort_columns, random_state=None,
			holdout_ratio=None, holdout_count=None, test_ratio=None, test_count=None,
			min_training_count=None, min_training_ratio=None
	):
		"""
		:type sort_columns: NoneType or list[str] or str
		:type random_state: NoneType or int or float
		"""
		if sort_columns is None:
			raise MissingArgumentError('sort_columns should be provided!')

		super().__init__(
			id_columns=None, random_state=random_state, sort_columns=sort_columns,
			holdout_ratio=holdout_ratio, holdout_count=holdout_count, test_ratio=test_ratio, test_count=test_count,
			min_training_count=min_training_count, min_training_ratio=min_training_ratio
		)
