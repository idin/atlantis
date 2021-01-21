from ...exceptions import FunctionNotImplementedError
from pandas import DataFrame

def get_display_function():
	try:
		from IPython.core.display import display
		return display
	except ImportError:
		return False


class DataContainer:
	def __init__(self, data, x_columns=None, y_column=None):
		self._data = data
		self._x_columns = x_columns
		self._y_column = y_column

	@property
	def data(self):
		"""
		:rtype: DataFrame
		"""
		return self._data

	def _repr_pretty_(self, p, cycle):
		if cycle:
			p.text(repr(self))
		else:
			self.display(p=p)

	def display(self):
		raise FunctionNotImplementedError('display function is not implemented for DataContainer!')