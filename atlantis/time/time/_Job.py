from .get_elapsed import get_elapsed
from .get_time import get_now


class Job:
	def __init__(self):
		self._start = None
		self._end = None

	def start(self):
		if self._start is None:
			self._start = get_now()
		else:
			raise RuntimeError('Job has already started!')

	def end(self):
		if self._start is None:
			raise RuntimeError('Job has not started yet!')
		elif self._end is None:
			self._end = get_now()
		else:
			raise RuntimeError('Job has already ended!')

	@property
	def start_time(self):
		if self._start is None:
			raise RuntimeError('Job has not started yet!')
		return self._start

	@property
	def end_time(self):
		if self._end is None:
			raise  RuntimeError('Job has not ended yet!')
		return self._end

	def get_elapsed(self, unit='ms'):
		return get_elapsed(start=self.start_time, end=self.end_time, unit=unit)
