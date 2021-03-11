import traceback
from ..time import get_elapsed, get_now


class TaskException:
	def __init__(self, task_id, exception, trace, worker_id):
		self._task_id = task_id
		self._exception = exception
		self._trace = trace
		self._worker_id = worker_id
		self._time = get_now()

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
		return self._trace

	@property
	def task_id(self):
		return self._task_id

	@property
	def time(self):
		return self._time

	@property
	def record(self):
		return {
			'task_id': self.task_id,
			'time': self.time,
			'exception': self.exception
		}


class Task:
	def __init__(self, function, task_id, args, kwargs, time_unit, cpu_count):
		self._function = function
		self._id = task_id
		self._args = args
		self._kwargs = kwargs
		self._time_unit = time_unit
		self._creation_time = get_now()
		self._cpu_count = cpu_count

	@property
	def id(self):
		return self._id

	@property
	def creation_time(self):
		return self._creation_time

	@property
	def time_unit(self):
		return self._time_unit

	@property
	def cpu_count(self):
		return self._cpu_count

	def do(self, worker_id=None):
		"""
		:rtype: Outcome
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
			return TaskException(task_id=self.id, exception=exception, trace=trace, worker_id=worker_id)

		else:
			end_time = get_now()
			return Outcome(
				task=self, result=result, worker_id=worker_id,
				start_time=start_time, end_time=end_time
			)


class Outcome:
	def __init__(self, task, result, start_time, end_time, worker_id):
		"""
		:type task: Task
		:param result:
		:param start_time:
		:param end_time:
		:param worker_id:
		"""
		self._result = result
		self._creation_time = task.creation_time
		self._time_unit = task.time_unit
		self._task_id = task.id
		self._start_time = start_time
		self._end_time = end_time
		self._elapsed = get_elapsed(start=self._start_time, end=self._end_time, unit=self._time_unit)

		self._worker_id = worker_id
		self._signature = {'function_name': task._function.__name__, 'elapsed': self.elapsed}

		if task._args is not None:
			for i, arg in enumerate(task._args):
				if isinstance(arg, (int, float, str)):
					self._signature[f'arg_{i + 1}'] = arg
		if task._kwargs is not None:
			for key, value in task._kwargs.items():
				if isinstance(value, (int, float, str)):
					self._signature[key] = value

	@property
	def task_id(self):
		return self._task_id

	@property
	def elapsed(self):
		return self._elapsed

	@property
	def timestamp_record(self):
		return {
			'task_id': self._task_id,
			'worker_id': self._worker_id,
			'creation_time': self._creation_time,
			'start_time': self._start_time,
			'end_time': self._end_time,
			'elapsed': self.elapsed
		}

	@property
	def signature(self):
		return self._signature

	@property
	def result(self):
		return self._result
