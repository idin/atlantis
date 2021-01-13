from pandas import DataFrame
from ...hash import hash_object
from ...exceptions import MissingColumnsError, MissingRowsError
from ...collections import OrderedSet
from ._get_rows import get_rows, get_columns


DATA_STATE_ATTRIBUTES = ['_columns', '_rows', '_original_data']


class DataSlice:
	def __init__(self, data, columns=None, rows=None):
		"""
		:type data: DataFrame or DataSlice
		"""
		self._parent = data
		if isinstance(self.parent, self.__class__):
			self._original_data = self.parent.original_data
		else:
			self._original_data = data

		if columns is None:
			self._columns = get_columns(self.parent)
		else:
			columns = OrderedSet(columns)
			parent_columns = get_columns(self.parent)
			if columns <= parent_columns:
				self._columns = columns
			else:
				raise MissingColumnsError(f'missing columns: {columns - parent_columns}')

		if rows is None:
			self._rows = get_rows(self.parent)
		else:
			rows = OrderedSet(rows)
			parent_rows = get_rows(self.parent)
			if rows <= parent_rows:
				self._rows = rows
			else:
				raise MissingRowsError(f'missing rows: {rows - parent_rows}')

	@property
	def parent(self):
		"""
		:rtype: DataFrame or DataSlice
		"""
		return self._parent

	@property
	def original_data(self):
		return self._original_data

	def __getstate__(self):
		return {key: getattr(self, key) for key in DATA_STATE_ATTRIBUTES}

	def __setstate__(self, state):
		for key, value in state.items():
			setattr(self, key, value)

	def get_data_hash(self):
		return hash_object(self.data)

	def __hashkey__(self):
		return self.original_data, self.columns, self.rows

	@property
	def data(self):
		"""
		:rtype: DataFrame
		"""
		return self.original_data[self.columns].iloc[self.rows]

	@property
	def columns(self):
		"""
		:rtype: list[str]
		"""
		return self._columns

	@property
	def rows(self):
		"""
		:rtype: list[int]
		"""
		return self._rows

	@property
	def num_columns(self):
		return self.shape[1]

	@property
	def num_rows(self):
		return self.shape[0]

	@property
	def shape(self):
		return self.data.shape

	def __repr__(self):
		return f'<Data: {self.num_rows}x{self.num_columns}>'
