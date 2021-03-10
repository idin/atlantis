from pandas.api.types import is_numeric_dtype


def get_numeric_columns(data):
	return [col for col in data.columns if is_numeric_dtype(data[col])]


def get_na_columns(data, only_numeric=True):
	if only_numeric:
		data = data[get_numeric_columns(data=data)]

	d = dict(data.isnull().any(axis=0))
	return [k for k, v in d.items() if v]


def fill_na_and_flag(data, fill_with='median'):
	data = data.copy()
	na_columns = get_na_columns(data=data, only_numeric=True)

	for column in na_columns:
		na_column = f'{column}_is_null'
		data[na_column] = data[column].isnull() * 1
		if fill_with == 'median':
			replacement = data[column].median()
		elif fill_with == 'mean':
			replacement = data[column].mean()
		else:
			raise ValueError(f'unknown option fill_with="{fill_with}"')
		data[column].fillna(replacement, inplace=True)

	return data


class NaFillerFlagger:
	def __init__(self, fill_with='median'):
		fill_with = fill_with.lower()
		if fill_with not in ['mean', 'median']:
			raise ValueError(f'unknown option fill_with="{fill_with}"')
		self._fill_with = fill_with
		self._na_columns = None

	def fit(self, X):
		self._na_columns = get_na_columns(data=X, only_numeric=True)

	def transform(self, X):
		data = X.copy()
		for col in self._na_columns:
			na_column = f'{col}_is_null'
			data[na_column] = data[col].isnull() * 1
			if self._fill_with == 'median':
				replacement = data[col].median()
			elif self._fill_with == 'mean':
				replacement = data[col].mean()
			else:
				raise ValueError(f'unknown option fill_with="{self._fill_with}"')
			data[col].fillna(replacement, inplace=True)

		for col in X.columns:
			if col not in self._na_columns and data[col].isnull().any():
				if self._fill_with == 'median':
					replacement = data[col].median()
				elif self._fill_with == 'mean':
					replacement = data[col].mean()
				else:
					raise ValueError(f'unknown option fill_with="{self._fill_with}"')
				data[col].fillna(replacement, inplace=True)
		return data

	def fit_transform(self, X):
		self.fit(X=X)
		return self.transform(X=X)
