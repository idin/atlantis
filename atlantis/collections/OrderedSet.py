# source: https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
import collections


class OrderedSet(collections.OrderedDict, collections.MutableSet):
	def __init__(self, *args):
		super().__init__()
		if len(args) == 1:
			try:
				for element in args[0]:
					self.add(element)
			except TypeError:
				if args[0] is not None:
					self.add(args[0])
		else:
			for element in args:
				self.add(element)

	def update(self, *args, **kwargs):
		if kwargs:
			raise TypeError("update() takes no keyword arguments")

		for s in args:
			for e in s:
				self.add(e)

	def add(self, elem):
		self[elem] = None

	def discard(self, elem):
		self.pop(elem, None)

	def __le__(self, other):
		return all(e in other for e in self)

	def __lt__(self, other):
		return self <= other and self != other

	def __ge__(self, other):
		return all(e in self for e in other)

	def __gt__(self, other):
		return self >= other and self != other

	def __repr__(self):
		return 'OrderedSet([%s])' % (', '.join(map(repr, self.keys())))

	def __str__(self):
		return '{%s}' % (', '.join(map(repr, self.keys())))

	def __sub__(self, other):
		return super().__sub__(other)

	def __and__(self, other):
		return super().__and__(other)

	def __isub__(self, other):
		return super().__isub__(other)

	def __iand__(self, other):
		return super().__iand__(other)

	def __xor__(self, other):
		return super().__xor__(other)

	def __ixor__(self, other):
		return super().__ixor__(other)

	def __or__(self, other):
		return super().__or__(other)

	difference = __sub__
	difference_update = __isub__
	intersection = __and__
	intersection_update = __iand__
	issubset = __le__
	issuperset = __ge__
	symmetric_difference = __xor__
	symmetric_difference_update = __ixor__
	union = __or__
