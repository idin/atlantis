from .SingleColumnImputer import SingleColumnImputer, ROW_NUM_COL
from pandas.api.types import is_numeric_dtype
from ...time.progress import ProgressBar
from pandas import DataFrame
from .exceptions import MissingModelError, IncludeExcludeClashError
from .exceptions import InputError
from copy import deepcopy


def get_columns_with_missing_values(data):
	"""
	:type data: DataFrame
	:rtype: list[str]
	"""
	return data.columns[data.isna().any()].tolist()


class Imputer:
	def __init__(
			self, regressor=None, classifier=None, impute_columns=None,
			exclude_columns=None, include_columns=None, echo=1
	):
		"""
		:param LinearRegression regressor: model to be used for imputation of numerical columns
		:param DecisionTreeClassifier or HyperModel classifier: model to be used for imputation of categorical columns
		:param NoneType or list[str] or str impute_columns: column(s) to be imputed, if None, all columns with missing values will be considered
		:param NoneType or list[str] or str exclude_columns: columns to be excluded from the model
		:param NoneType or list[str] or str include_columns: if not None, only these columns of the dataset will be used
		"""
		if regressor is None and classifier is None:
			raise MissingModelError('at least one of regressor or classifier should be provided!')

		self._regressor = regressor
		self._classifier = classifier

		if isinstance(impute_columns, str):
			impute_columns = [impute_columns]
		if isinstance(exclude_columns, str):
			exclude_columns = [exclude_columns]
		if isinstance(include_columns, str):
			include_columns = [include_columns]

		self._impute_columns = impute_columns
		self._exclude_columns = exclude_columns or []
		self._include_columns = include_columns

		if exclude_columns is not None and include_columns is not None:
			columns_in_both = [column for column in exclude_columns if column in include_columns]
			if len(columns_in_both) > 0:
				raise IncludeExcludeClashError(f'You cannot include and exclude a column! These are in both: {columns_in_both}')

		self._imputers = {}
		self._echo = echo
		self._logs = ''

	def fit(self, X):
		"""
		:param DataFrame X: the dataframe to be imputed
		:rtype: DataFrame
		"""
		# find columns that are only in X and are only numeric and are only in include
		ok_columns = [column for column in X.columns if is_numeric_dtype(X[column])]
		if self._include_columns is not None:
			ok_columns = [x for x in ok_columns if x in self._include_columns]
		if self._exclude_columns is not None:
			ok_columns = [x for x in ok_columns if x not in self._exclude_columns]

		impute_columns = self._impute_columns
		if impute_columns is None:
			impute_columns = get_columns_with_missing_values(data=X)
			if self._include_columns is not None:
				impute_columns = [col for col in impute_columns if col in self._include_columns]
			elif self._exclude_columns is not None:
				impute_columns = [col for col in impute_columns if col not in self._exclude_columns]

		if not isinstance(impute_columns, (list, str)) and impute_columns is not None:
			raise InputError(f'columns is of type {type(impute_columns)}')
		elif isinstance(impute_columns, str):
			impute_columns = [impute_columns]

		progress_bar = ProgressBar(total=len(impute_columns), echo=self._echo)
		progress = 0

		for column in impute_columns:
			progress_bar.show(amount=progress, text=f'fitting imputer for column: "{column}"')
			if is_numeric_dtype(X[column]):
				model = self._regressor
				column_type = 'numerical'
			else:
				model = self._classifier
				column_type = 'nonnumerical'

			if model is None:
				raise RuntimeError(f'no model for {column} which is {X[column].dtype}')

			model = deepcopy(model)
			single_column_imputer = SingleColumnImputer(
				estimator=model, column=column, column_type=column_type,
				use_columns=ok_columns
			)

			single_column_imputer.fit(X)
			self._imputers[column] = single_column_imputer

			model_name = model.__class__.__name__
			if model_name[0].lower() in ['a', 'e', 'i', 'o']:
				a = 'an'
			else:
				a = 'a'
			self._logs += f'fitted {a} "{model_name}" imputer for the {column_type} column "{column}".\n'
			progress += 1
		progress_bar.show(amount=progress, text=f'fitting imputers complete')
		self._logs += '\n'

	@property
	def single_column_imputers(self):
		"""
		:rtype: dict[str, SingleColumnImputer]
		"""
		return self._imputers

	def transform(self, X):
		"""
		:type X: DataFrame
		:rtype: DataFrame
		"""

		X = X.copy()
		X[ROW_NUM_COL] = range(X.shape[0])
		progress_bar = ProgressBar(total=len(self.single_column_imputers), echo=self._echo)
		progress = 0
		for column, imputer in self.single_column_imputers.items():
			progress_bar.show(amount=progress, text=f'imputating column "{column}"')
			X[column] = imputer.impute_column(data=X)
			self._logs += f'imputed column "{column}".\n'
			progress += 1
		progress_bar.show(amount=progress, text='imputatation complete')
		self._logs += '\n'
		return X.drop(columns=ROW_NUM_COL)

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)
