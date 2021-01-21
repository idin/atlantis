
from ...validation import Validation
from .._DataSlice import TrainingTestSlice
from ._LearningProject import LearningProject
from pandas import DataFrame


class CrossValidationProject(LearningProject):
	def add_validation(
			self, data, validation,
			id_prefix='fold_',
			overwrite=False, random_state=None
	):
		"""

		:param data:
		:type  data: DataFrame

		:param validation:
		:type  validation: Validation

		:param processor:

		:param id_prefix:
		:type  id_prefix: str

		:type  overwrite: bool
		:type  random_state: int
		"""
		container = validation.split(data=data, random_state=random_state)
		data_id = self.name
		self.processor.add_data(data_id=data_id, data=container.data, overwrite=overwrite)

		for i, fold in enumerate(container.folds):
			training_test_slice_id = f'{id_prefix}{i + 1}'
			training_test_slice = TrainingTestSlice(
				data_id=data_id, training_indices=fold.training_indices, test_indices=fold.test_indices,
				columns=None
			)

			self.add_training_test_slice(
				training_test_slice_id=training_test_slice_id,
				training_test_slice=training_test_slice,
				data=None,
				overwrite=overwrite
			)
