import numpy as np
import pandas as pd
from pandas import DataFrame
from ...time.progress import ProgressBar


class OneHotEncoder:
	def __init__(
			self, top=10, rank_method='first', encode_na=False, replacement='other',
			include=None, exclude=None, lowercase=True
	):
		self._column_values = {}
		self._top = top
		self._method = rank_method
		self._encode_na = encode_na
		self._replacement = replacement
		self._include = include
		self._exclude = exclude
		self._lowercase = lowercase
		self._one_hot_columns = None
		self._encoded_columns = None

	@property
	def encoded_columns(self):
		"""
		:rtype: list[str]
		"""
		return self._encoded_columns

	@property
	def one_hot_columns(self):
		"""
		:rtype: list[str]
		"""
		return self._one_hot_columns

	def fit(self, data, echo=0):
		echo = max(0, echo)
		result = data.copy()
		one_hot_columns = []
		non_numeric_cols = data.select_dtypes(exclude=['bool', 'number', 'datetime64', 'datetime']).columns

		if self._include is not None:
			non_numeric_cols = [col for col in non_numeric_cols if col in self._include]
		if self._exclude is not None:
			non_numeric_cols = [col for col in non_numeric_cols if col not in self._exclude]

		progress_bar = ProgressBar(total=len(non_numeric_cols))
		progress_amount = 0
		self._encoded_columns = []
		for col_name in non_numeric_cols:
			if echo:
				progress_bar.show(amount=progress_amount, text=f'DM training dummies for {col_name}')
			progress_amount += 1
			try:
				temp_data = data[[col_name]].copy()
				if self._lowercase:
					temp_data[col_name] = temp_data[col_name].str.lower()
				temp_data['count'] = 1
				counts = temp_data.groupby(col_name).sum().reset_index(drop=False)

				if self._top is not None:
					top_counts = counts.sort_values(by=col_name, ascending=False).head(self._top)
					only_include = set(top_counts[col_name])
					temp_data[col_name] = np.where(
						temp_data[col_name].isin(only_include), temp_data[col_name], self._replacement
					)

				else:
					only_include = set(counts[col_name])

				dummies = pd.get_dummies(
					data=temp_data[[col_name]],
					prefix=col_name, prefix_sep='_', dummy_na=self._encode_na, sparse=False
				)

				dummies = dummies[[col for col in dummies if not dummies[col].nunique() == 1]]

				result = pd.concat([result, dummies], axis=1)
				one_hot_columns += list(dummies.columns)

				self._column_values[col_name] = only_include
				self._encoded_columns.append(col_name)
			except Exception as e:
				print(f'exception being raised for column: {col_name}')
				raise e
		self._one_hot_columns = one_hot_columns
		if echo:
			progress_bar.show(amount=progress_amount, text=f'DM trained dummies for {self._encoded_columns}')
		return result

	def transform(self, data, echo=0):
		return self.encode(data=data, echo=echo)

	def fit_transform(self, data, echo=0):
		self.fit(data=data, echo=echo)
		return self.transform(data=data, echo=echo)

	def encode(self, data, drop_encoded=True, echo=0):
		echo = max(0, echo)
		result = data.copy()
		progress_bar = ProgressBar(total=len(self._column_values))
		progress_amount = 0
		for col_name, only_include in self._column_values.items():
			if echo:
				progress_bar.show(amount=progress_amount, text=f'DM creating dummies for {col_name}')
			progress_amount += 1
			temp_data = result[[col_name]].copy()
			if self._lowercase:
				temp_data[col_name] = temp_data[col_name].str.lower()
			temp_data[col_name] = np.where(
				temp_data[col_name].isin(only_include), temp_data[col_name], self._replacement
			)
			dummies = pd.get_dummies(
				data=temp_data[[col_name]],
				prefix=col_name, prefix_sep='_', dummy_na=self._encode_na, sparse=True
			)
			result = pd.concat([result, dummies], axis=1)
		for col_name in self._one_hot_columns:
			if col_name not in result.columns:
				result[col_name] = 0
		if echo:
			progress_bar.show(amount=progress_amount, text=f'DM created dummies for {self._encoded_columns}')

		if drop_encoded:
			result = result.drop(columns=self.encoded_columns)

		extra_columns = [
			x for x in result.columns
			if x not in self.encoded_columns + self.one_hot_columns and x not in data.columns
		]
		if len(extra_columns) > 0:
			result = result.drop(columns=extra_columns)

		return result

	def untransform(self, data, echo=0):
		echo = max(0, echo)
		result = data.copy()
		progress_bar = ProgressBar(total=len(self._column_values))
		progress_amount = 0
		# TODO

	@classmethod
	def get_encoder_and_encoded(cls, data, echo=0, **kwargs):
		"""
		:rtype: tuple[OneHotEncoder, DataFrame]
		"""
		one_hot_encoder = cls(**kwargs)
		data_encoded = one_hot_encoder.fit_transform(data=data, echo=echo)
		return one_hot_encoder, data_encoded


def get_one_hot_encoder_and_encoded(data, echo=0, **kwargs):
	return OneHotEncoder.get_encoder_and_encoded(data=data, echo=echo, **kwargs)
