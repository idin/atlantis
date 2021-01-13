from pandas import DataFrame
from ...collections import cross_lists


class Polynomial:
	def __init__(self, degree=2, sep='.'):
		self._degree = degree
		self._columns = None
		self._sep = sep

	@property
	def degree(self):
		"""
		:rtype: int
		"""
		return self._degree

	def fit(self, data):
		"""
		:type data: DataFrame
		"""
		if self._columns is not None:
			raise RuntimeError('polynomial has already been fit!')

		self._columns = list(data.columns)

	def get_columns(self, degree):
		combinations = cross_lists([self._columns] * degree)
		ordered_combinations = [sorted(x) for x in combinations]
		unique_combinations = []
		for combination in ordered_combinations:
			if combination not in unique_combinations:
				unique_combinations.append(combination)
		return {self.get_column_name(combination): combination for combination in unique_combinations}

	def get_column_name(self, combination):
		# count repetition of elements
		counts = {x: combination.count(x) for x in set(combination)}
		return self._sep.join([
			str(column) if counts[column] == 1 else f'{column}{counts[column]}'
			for column in sorted(set(combination))
		])

	def transform(self, data):
		"""
		:type data: DataFrame
		:rtype: DataFrame
		"""
		data = data.copy()
		for degree_minus_one in range(1, self._degree):
			for column_name, combination in self.get_columns(degree=degree_minus_one + 1).items():
				data[column_name] = 1
				for column in combination:
					data[column_name] = data[column_name] * data[column]
		return data

	def fit_transform(self, data):
		"""
		:type data: DataFrame
		:rtype: DataFrame
		"""
		self.fit(data=data)
		return self.transform(data=data)
