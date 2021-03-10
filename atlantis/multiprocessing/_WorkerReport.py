from ..time import get_now


class WorkerReport:
	def __init__(self, worker_id):
		self._worker_id = worker_id
		self._start_time = get_now()
		self._end_time = None
		self._tasks = set()

	def add_task_id(self, task_id):
		self._tasks.add(task_id)

	def end(self):
		self._end_time = get_now()

	def __repr__(self):
		return f'Worker {self._worker_id} | {self._start_time}-{self._end_time} | {len(self._tasks)} tasks done.'

	@property
	def record(self):
		return {
			'worker_id': self._worker_id,
			'start_time': self._start_time,
			'end_time': self._end_time,
			'task_count': len(self._tasks)
		}


class SupervisorReport:
	def __init__(self, supervisor_id):
		self._supervisor_id = supervisor_id
		self._start_time = get_now()
		self._end_time = None
		self._logs = []

	def log(self, string):
		self._logs.append((get_now(), string))

	def __repr__(self):
		return f'Supervisor {self._supervisor_id} | {self._start_time}-{self._end_time} | {len(self._logs)} tasks done.'

	@property
	def records(self):
		return [
			{'time': log[0], 'event': log[1]}
			for log in self._logs
		]

	def end(self):
		self._end_time = get_now()