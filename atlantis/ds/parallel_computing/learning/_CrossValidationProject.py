
from ...validation import Validation
from .._DataSlice import TrainingTestSlice
from ._LearningProject import LearningProject
from ._LearningTask import LearningTask
from pandas import DataFrame


class CrossValidationProject(LearningProject):
	def __init__(
			self, name, y_column, problem_type, x_columns=None,
			time_unit='ms', evaluation_function=None, main_metric=None,
			lowest_is_best=None, best_score=None,
			scoreboard=None, processor=None
	):
		super().__init__(
			name=name, y_column=y_column, problem_type=problem_type, x_columns=x_columns,
			time_unit=time_unit, evaluation_function=evaluation_function, main_metric=main_metric,
			lowest_is_best=lowest_is_best, best_score=best_score, scoreboard=scoreboard, processor=processor
		)
		self._validation_holdout_slice_id = None
		self._full_data_slice_id = None

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
			training_test_slice_id = f'{self.name}_{id_prefix}{i + 1}'
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

		if container.holdout_indices is not None:
			validation_holdout_slice = TrainingTestSlice(
				data_id=data_id, training_indices=container.validation_indices,
				test_indices=container.holdout_indices, columns=None
			)
			self._add_validation_holdout_slice(
				validation_holdout_slice_id=f'{self.name}_validation_holdout',
				validation_holdout_slice=validation_holdout_slice,
				data=None,
				overwrite=overwrite
			)

		# add full data
		full_data_slice = TrainingTestSlice(
			data_id=data_id, training_indices=None,
			test_indices=None,
			columns=None
		)
		self._add_full_data_slice(
			full_data_slice_id=f'{self.name}_full_data',
			full_data_slice=full_data_slice, data=None, overwrite=overwrite
		)

	def _add_validation_holdout_slice(
			self, validation_holdout_slice, data=None, validation_holdout_slice_id=None, overwrite=False
	):
		if validation_holdout_slice_id is None:
			validation_holdout_slice_id = f'{self.name}_validation_holdout'

		self._validation_holdout_slice_id = validation_holdout_slice_id

		self.add_training_test_slice(
			training_test_slice_id=validation_holdout_slice_id,
			training_test_slice=validation_holdout_slice,
			data=data,
			overwrite=overwrite,
			add_to_training_test_ids=False
		)

	def _add_full_data_slice(
			self, full_data_slice, data=None, full_data_slice_id=None, overwrite=False
	):
		if full_data_slice_id is None:
			full_data_slice_id = f'{self.name}_full_data'

		self._full_data_slice_id = full_data_slice_id

		self.add_training_test_slice(
			training_test_slice_id=full_data_slice_id,
			training_test_slice=full_data_slice,
			data=data,
			overwrite=overwrite,
			add_to_training_test_ids=False
		)

	def get_holdout_results(self):
		best_estimator = self.get_best_estimator()
		task = LearningTask(
			project_name=self.name, estimator_class=best_estimator['class'],
			estimator_name=best_estimator['name'],
			estimator_id=best_estimator['id'],
			estimator_arguments=best_estimator['arguments'],
			training_test_slice_id=self._validation_holdout_slice_id,
			y_column=self.y_column, x_columns=self.x_columns,
			evaluation_function=self.evaluation_function
		)
		task.do(namespace=self.processor.namespace, worker_id='main', return_predictions=True)
		return task

	def get_complete_model(self):
		best_estimator = self.get_best_estimator()
		task = LearningTask(
			project_name=self.name, estimator_class=best_estimator['class'],
			estimator_name=best_estimator['name'],
			estimator_id=best_estimator['id'],
			estimator_arguments=best_estimator['arguments'],
			training_test_slice_id=self._full_data_slice_id,
			y_column=self.y_column, x_columns=self.x_columns,
			evaluation_function=self.evaluation_function
		)
		task.do(namespace=self.processor.namespace, worker_id='main', return_predictions=True)
		return task
