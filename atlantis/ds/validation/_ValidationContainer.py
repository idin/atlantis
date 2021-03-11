from pandas import DataFrame
from ._DataContainer import get_display_function, DataContainer
from ._TrainingTestContainer import TrainingTestContainer
from ...time.progress import ProgressBar
from ._train_and_test import train_and_test


class ValidationContainer(DataContainer):
	def __init__(
			self, data, validation_indices, holdout_indices, folds, x_columns=None, y_column=None,
			sort_columns=None
	):
		super().__init__(data=data, x_columns=x_columns, y_column=y_column, sort_columns=sort_columns)
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

	@property
	def num_folds(self):
		return len(self._folds)

	def get_fold(self, n):
		"""
		:param n: a number between 1 and number of folds (it doesn't start at zero)
		:rtype: TrainingTestContainer
		"""
		return self.folds[n - 1]

	def evaluate_fold(
			self, fold_n, estimator, evaluation_type,
			estimator_id=None, x_columns=None, y_column=None,
			ignore_missing_y=False, ignore_missing_x=False,
			replace_missing_x='mean'
	):
		y_column = y_column or self._y_column
		if y_column is None:
			raise ValueError(f'y_column should be provided')
		x_columns = x_columns or self._x_columns
		if x_columns is None:
			x_columns = [col for col in self.columns if col != y_column]
		training_test_container = self.get_fold(n=fold_n)
		training_data = training_test_container.training_data
		test_data = training_test_container.test_data

		result = train_and_test(
			estimator=estimator, evaluation_type=evaluation_type,
			training_data=training_data, test_data=test_data,
			x_columns=x_columns, y_column=y_column,
			sort_columns=self._sort_columns,
			ignore_missing_y=ignore_missing_y,
			ignore_missing_x=ignore_missing_x,
			replace_missing_x=replace_missing_x
		)
		if estimator_id is not None:
			result = {'estimator_id': estimator_id, **result}

		return result

	def _generate_evaluation(self, estimators, evaluation_type, x_columns=None, y_column=None):
		"""
		a generator for the evaluate function
		:param estimators:
		:param evaluation_type:
		:param x_columns:
		:param y_column:
		:param aggregate:
		:rtype: Generator[dict] or list[dict]
		"""
		if not isinstance(estimators, (list, tuple, dict)):
			estimators = [estimators]

		if isinstance(estimators, (list, tuple)):
			estimators = {(i + 1): x for i, x in enumerate(estimators)}

		y_column = y_column or self._y_column
		if y_column is None:
			raise ValueError(f'y_column should be provided')
		x_columns = x_columns or self._x_columns
		if x_columns is None:
			x_columns = [col for col in self.columns if col != y_column]

		for i, fold in enumerate(self.folds):
			for estimator_id, estimator in estimators.items():
				yield {
					'kwargs': {
						'training_data': fold.training_data,
						'test_data': fold.test_data,
						'estimator': estimator,
						'evaluation_type': evaluation_type,
						'x_columns': x_columns,
						'y_column': y_column,
						'sort_columns': self._sort_columns
					},
					'estimator_id': estimator_id,
					'estimator_name': estimator.__class__.__name__,
					'fold': i + 1,
				}

	def _evaluate(self, estimators, evaluation_type, x_columns=None, y_column=None, aggregate=False, echo=1):
		records = []
		progress_bar = ProgressBar(total=len(estimators) * len(self.folds) + 1, echo=echo)
		progress_amount = 0

		for dictionary in self._generate_evaluation(
				estimators=estimators, evaluation_type=evaluation_type, x_columns=x_columns, y_column=y_column
		):
			fold_number = dictionary['fold']
			estimator_id = dictionary['estimator_id']
			progress_bar.show(amount=progress_amount, text=f'evaluating fold {fold_number} with estimator {estimator_id}')
			evaluation = train_and_test(**dictionary['kwargs'])
			records.append({
				'fold': fold_number,
				'estimator_id': estimator_id,
				'estimator_name': dictionary['estimator_name'],
				**evaluation
			})
			progress_amount += 1

		progress_bar.show(amount=progress_amount, text=f'aggregating results!')
		df = DataFrame.from_records(records)
		if aggregate:
			df = df.drop(columns='fold').groupby(
				['estimator_id', 'estimator_name']
			).agg(['mean', 'median', 'min', 'max', 'std'])
		progress_amount += 1
		progress_bar.show(amount=progress_amount, text=f'evaluation complete!')

		return df

	def evaluate_regression(self, estimators, x_columns=None, y_column=None, aggregate=False, echo=1):
		"""
		evaluates one or more regressors on all the folds and returns a dataframe of the results
		:param estimators: one or more regressors
		:param x_columns: the x columns (independent variables)
		:param y_column: the y column (dependent variable)
		:param aggregate: if True, the evaluation metrics will be aggregated over the folds using mean, median, std, min, max
		:param echo: to show progress or not
		:rtype: DataFrame
		"""
		return self._evaluate(
			evaluation_type='regression',
			estimators=estimators, x_columns=x_columns, y_column=y_column, echo=echo, aggregate=aggregate
		)

	def evaluate_classification(self, estimators, x_columns=None, y_column=None, aggregate=False, echo=1):
		"""
		evaluates one or more classifiers on all the folds and returns a dataframe of the results
		:param estimators: one or more classifiers
		:param x_columns: the x columns (independent variables)
		:param y_column: the y column (dependent variable)
		:param aggregate: if True, the evaluation metrics will be aggregated over the folds using mean, median, std, min, max
		:param echo: to show progress or not
		:rtype: DataFrame
		"""
		return self._evaluate(
			evaluation_type='classification',
			estimators=estimators, x_columns=x_columns, y_column=y_column, echo=echo, aggregate=aggregate
		)

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
