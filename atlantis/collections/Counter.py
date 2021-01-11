from ._make_hashable import make_hashable
import heapq as _heapq


class Counter:
	'''
	similar to the Counter class from collections but with the ability to count objects that are not hashable
	'''
	def __init__(self, l):
		self._counts = {}
		self._values = {}
		for x in l:
			hashable = make_hashable(x)
			if hashable in self._counts:
				self._counts[hashable] += 1
				self._values[hashable] = x
			else:
				self._counts[hashable] = 1
				self._values[hashable] = x

	def items(self):
		return [(self._values[key], count) for key, count in self._counts.items()]

	def most_common(self, n=None):
		'''
		List the n most common elements and their counts from the most
		common to the least.  If n is None, then list all element counts.

		>>> Counter('abcdeabcdabcaba').most_common(3)
		[('a', 5), ('b', 4), ('c', 3)]

		'''
		if n is None:
			sorted_counts = sorted(self._counts.items(), key=lambda x: x[1], reverse=True)
			return [(self._values[hashable], count) for hashable, count in sorted_counts]

		else:
			sorted_counts = _heapq.nlargest(n, self._counts.items(), key=lambda x: x[1])
			return [(self._values[hashable], count) for hashable, count in sorted_counts]

	def __repr__(self):
		string = ', '.join(f'{x}: {y}' for x, y in self.most_common())
		return 'Counter: {' + string + '}'
