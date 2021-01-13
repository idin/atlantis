from pandas import DataFrame
from numpy import isnan


class Normalizer:
	def __init__(self, ignore_columns=None):
		self._columns = None
		self._means = None
		self._standard_deviations = None

		if isinstance(ignore_columns, str):
			ignore_columns = [ignore_columns]
		self._ignore_columns = ignore_columns or []

	def __getstate__(self):
		return {
			'_columns': self._columns, '_means': self._means, '_standard_deviations': self._standard_deviations,
			'_ignore_columns': self._ignore_columns,
		}

	def __setstate__(self, state):
		for key, value in state.items():
			setattr(self, key, value)

	def fit(self, X, y=None):
		"""
		:type X: DataFrame
		"""
		if self._means is not None or self._standard_deviations is not None:
			raise RuntimeError('normalizer is already fit!')

		self._means = {
			column: X[column].mean(skipna=True)
			for column in X.columns if column not in self._ignore_columns
		}

		self._standard_deviations = {
			column: X[column].std(skipna=True)
			for column in X.columns if column not in self._ignore_columns
		}

		nan_values = []
		for column, m in self._means.items():
			if isnan(m) or isnan(self._standard_deviations[column]):
				nan_values.append(str(column))
		if len(nan_values) > 0:
			raise ValueError(f'{", ".join(nan_values)} produce nan values!')

	def transform(self, X):
		"""
		:type X: DataFrame
		:rtype: DataFrame
		"""
		if self._means is None or self._standard_deviations is None:
			raise RuntimeError('normalizer is not fit!')

		result = X.copy()

		for column in self._means.keys():
			if self._standard_deviations[column] != 0:
				result[column] = (result[column] - self._means[column]) / self._standard_deviations[column]
			else:
				result[column] = (result[column] - self._means[column])

		return result

	def fit_transform(self, X, y=None):
		self.fit(X=X, y=y)
		return self.transform(X=X)

	def untransform(self, X):
		"""
		:type X: DataFrame
		:rtype: DataFrame
		"""
		if self._means is None or self._standard_deviations is None:
			raise RuntimeError('normalizer is not fit!')

		result = X.copy()

		for column in self._means.keys():
			if self._standard_deviations[column] != 0:
				result[column] = result[column] * self._standard_deviations[column] + self._means[column]
			else:
				result[column] = result[column] + self._means[column]

		return result

	@property
	def parameters(self):
		"""
		:rtype: DataFrame
		"""
		return DataFrame.from_records([
			{'column': column, 'mean': self._means[column], 'sd': self._standard_deviations[column]}
			for column in self._means.keys()
		])

	@classmethod
	def get_normalizer_and_normalized(cls, data, **kwargs):
		"""
		:rtype: tuple[Normalizer, DataFrame]
		"""
		normalizer = cls(**kwargs)
		normalized = normalizer.fit_transform(X=data)
		return normalizer, normalized


def get_normalizer_and_normalized(data, **kwargs):
	"""
	:rtype: tuple[Normalizer, DataFrame]
	"""
	return Normalizer.get_normalizer_and_normalized(data=data, **kwargs)
