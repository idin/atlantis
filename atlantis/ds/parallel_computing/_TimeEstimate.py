class TimeEstimate:
	def __init__(self, count=0, total=0):
		self._total = total
		self._count = count

	def append(self, elapsed):
		self._total += elapsed
		self._count += 1

	def get_mean(self):
		return self._total / self._count

	@property
	def mean(self):
		return self.get_mean()

	def __repr__(self):
		return str(self.mean)

	def __add__(self, other):
		"""
		:type other: TimeEstimate
		:rtype: TimeEstimate
		"""
		return TimeEstimate(count=self._count + other._count, total=self._total + other._total)


class MissingTimeEstimate:
	def __eq__(self, other):
		return isinstance(other, MissingTimeEstimate)

	def __ne__(self, other):
		return not isinstance(other, MissingTimeEstimate)
