from ...time import get_elapsed
from datetime import datetime


class Task:
	def __init__(self, project_name, task_id):
		self._project_name = project_name
		self._status = 'new'
		self._worker_id = None
		self._error = None
		self._starting_time = None
		self._ending_time = None
		self._id = task_id

	def __str__(self):
		return f'{self.id} ({self._status})'

	def __repr__(self):
		return f'{self.__class__.__name__}: {self})'

	@property
	def id(self):
		if self._id is None:
			raise ValueError(f'task id is None!')
		return self._id

	@property
	def project_name(self):
		return self._project_name

	@property
	def starting_time(self):
		return self._starting_time

	@property
	def ending_time(self):
		return self._ending_time

	def start(self):
		self._status = 'started'
		self._starting_time = datetime.now()

	def do(self, data_namespace, worker_id):
		try:
			self.start()

			# placeholder for whatever needs to be done
			raise NotImplementedError(f'do() is not implemented for this task: {self.id}')

			self.end(worker_id=worker_id)
		except Exception as error:
			self.set_error(error=error)

	def end(self, worker_id):
		self._ending_time = datetime.now()
		self._status = 'done'
		self._worker_id = worker_id

	def get_elapsed(self, unit='ms'):
		if self._starting_time is not None and self._ending_time is not None:
			return get_elapsed(start=self.starting_time, end=self.ending_time, unit=unit)
		else:
			return None

	def set_error(self, error):
		self._error = error
		self._status = 'error'

	def is_done(self):
		return self._status == 'done'



