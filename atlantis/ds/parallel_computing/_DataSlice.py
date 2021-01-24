from pandas import DataFrame


class DataSlice:
	def __init__(self, data_id, indices, columns=None):
		self._data_id = data_id
		self._columns = columns
		self._indices = indices

	@property
	def data_id(self):
		return self._data_id

	def get_data(self, data):
		"""
		:type data: DataFrame
		:rtype: DataFrame
		"""
		if self._columns is not None:
			data = data[self._columns]
		if self._indices is not None:
			return data.iloc[self._indices]
		else:
			return data


class TrainingTestSlice:
	def __init__(self, data_id, training_indices, test_indices, columns=None):
		self._training_slice = DataSlice(data_id=data_id, indices=training_indices, columns=columns)
		self._test_slice = DataSlice(data_id=data_id, indices=test_indices, columns=columns)

	@property
	def data_id(self):
		return self._training_slice.data_id

	def get_training_data(self, data):
		return self._training_slice.get_data(data=data)

	def get_test_data(self, data):
		return self._test_slice.get_data(data=data)