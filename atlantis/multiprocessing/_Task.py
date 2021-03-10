import traceback
from ..time import get_elapsed, get_now


class TaskResult:
	def __init__(self, result, start_time, end_time, worker_id, task_id, time_unit):
		self._result = result
		self._start_time = start_time
		self._end_time = end_time
		self._task_id = task_id
		self._time_unit = time_unit
		self._worker_id = worker_id

	@property
	def elapsed(self):
		return get_elapsed(start=self._start_time, end=self._end_time, unit=self._time_unit)

	@property
	def timestamp_record(self):
		return {
			'task_id': self._task_id,
			'worker_id': self._worker_id,
			'start_time': self._start_time,
			'end_time': self._end_time,
			'elapsed': self.elapsed
		}


class TaskException:
	def __init__(self, exception, traceback, worker_id):
		self._exception = exception
		self._traceback = traceback
		self._worker_id = worker_id

	@property
	def exception(self):
		"""
		:rtype: Exception
		"""
		return self._exception

	@property
	def traceback(self):
		"""
		:rtype: str
		"""
		return self._traceback


class Task:
	def __init__(self, function, task_id, args, kwargs, time_unit):
		self._function = function
		self._id = task_id
		self._args = args
		self._kwargs = kwargs
		self._time_unit = time_unit

	@property
	def id(self):
		return self._id

	def do(self, worker_id=None):
		"""
		:rtype: TaskResult
		"""
		start_time = get_now()
		try:
			if self._args is None:
				if self._kwargs is None:
					result = self._function()
				else:
					result = self._function(**self._kwargs)
			else:
				if self._kwargs is None:
					result = self._function(*self._args)
				else:
					result = self._function(*self._args, **self._kwargs)
		except Exception as exception:
			trace = traceback.format_exc()
			print(f'task {self.id} exception')
			traceback.print_exc()
			return TaskException(exception=exception, traceback=trace, worker_id=worker_id)

		else:
			end_time = get_now()
			return TaskResult(
				result=result, start_time=start_time, end_time=end_time,
				task_id=self._id, time_unit=self._time_unit,
				worker_id=worker_id
			)