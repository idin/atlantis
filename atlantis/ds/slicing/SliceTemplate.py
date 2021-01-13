from .DataSlice import DataSlice
from pandas import DataFrame
from ...collections import OrderedSet


def get_and(x, y):
	if x is None:
		return y
	elif y is None:
		return x
	else:
		return x & y


def get_or(x, y):
	if x is None:
		return y
	elif y is None:
		return x
	else:
		return x | y


class SliceTemplate:
	def __init__(self, columns, rows):
		if columns is None:
			self._columns = None
		else:
			self._columns = OrderedSet(columns)

		if rows is None:
			self._rows = None
		else:
			self._rows = OrderedSet(rows)

	def get_slice(self, data):
		"""
		:type data: DataFrame
		:rtype: DataSlice
		"""
		return DataSlice(data=data, columns=self._columns, rows=self._rows)

	def __and__(self, other):
		return self.__class__(columns=get_and(self._columns, other._columns), rows=get_and(self._rows, other._rows))
