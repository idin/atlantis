from pandas import DataFrame
from ._DataContainer import get_display_function, DataContainer
from ._TrainingTestContainer import TrainingTestContainer


class ValidationContainer(DataContainer):
	def __init__(self, data, validation_indices, holdout_indices, folds, x_columns=None, y_column=None):
		super().__init__(data=data, x_columns=x_columns, y_column=y_column)
		self._validation_indices = validation_indices
		self._holdout_indices = holdout_indices
		self._folds = folds

	@property
	def validation_indices(self):
		return self._validation_indices

	@property
	def holdout_indices(self):
		return self._holdout_indices

	@property
	def validation(self):
		"""
		:rtype: DataFrame
		"""
		if self._validation_indices is None:
			return None
		else:
			return self._data.iloc[self._validation_indices]

	@property
	def holdout(self):
		"""
		:rtype: DataFrame
		"""
		if self._holdout_indices is None:
			return None
		else:
			return self._data.iloc[self._holdout_indices]

	@property
	def folds(self):
		"""
		:rtype: list[TrainingTestContainer]
		"""
		return self._folds

	def display(self, p=None):
		display = get_display_function()

		if len(self.folds) > 0:
			for i, fold in enumerate(self.folds):
				fold.display(p=p, prefix=f'Fold {i + 1} ', function=display)
			if display is False:
				print('Holdout:')
				print(self.holdout)
			else:
				print('Holdout:')
				display(self.holdout)
		else:
			if display is False:
				print('Validation:')
				print(self.validation)
				print('Holdout:')
				print(self.holdout)
			else:
				print('Validation:')
				display(self.validation)
				print('Holdout:')
				display(self.holdout)
