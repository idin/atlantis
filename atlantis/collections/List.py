from ._product import get_product
from ._duplicates import get_duplicates, remove_duplicates, has_duplicates
from ._flatten import flatten
from ._make_hashable import make_hashable
from .Counter import Counter


class List(list):
	def __init__(self, *args):
		if len(args) == 0:
			super().__init__()

		elif len(args) == 1:
			if isinstance(args[0], (list, tuple)):
				the_list = [
					self.__class__(element) if isinstance(element, list) and not isinstance(element, self.__class__)
					else element
					for element in args[0]
				]
				super().__init__(the_list)
			else:
				try:
					super().__init__(args[0])
				except TypeError:
					super().__init__([args[0]])

		else:
			the_list = [
				self.__class__(element) if isinstance(element, list) and not isinstance(element, self.__class__)
				else element
				for element in args
			]
			super().__init__(the_list)

	def __repr__(self):
		return f'List: [{", ".join(str(x) for x in self)}]'

	def __str__(self):
		return super().__str__()

	def __mul__(self, other):
		if isinstance(other, list):
			return self.__class__(get_product(self, other))
		else:
			return self * other

	def has_duplicates(self):
		"""
		returns True if list has duplicates
		:rtype: bool
		"""
		return has_duplicates(self)

	def remove_duplicates(self):
		"""
		removes the duplicates from list
		:rtype: List
		"""
		return self.__class__(remove_duplicates(self))

	def get_duplicates(self):
		"""
		returns duplicate values in list
		:rtype: List
		"""
		return self.__class__(get_duplicates(self))

	def flatten(self):
		"""
		flattens a list
		:rtype: List
		"""
		return self.__class__(flatten(self))

	def count(self, obj=None):
		"""
		if obj is None, counts the number of elements, otherwise counts the number of obj
		:rtype: List or int
		"""
		if obj is None:
			counter = Counter(self)
			return self.__class__(counter.items())
		else:
			return super().count(obj)

	def __hashkey__(self):
		return tuple([make_hashable(x) for x in self])
