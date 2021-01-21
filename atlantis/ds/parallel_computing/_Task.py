from ...time import get_elapsed
from ._get_data_from_namespace import get_data_from_namespace
from datetime import datetime
import traceback

class Task:
	def __init__(self, project_name, task_id):
		self._project_name = project_name
		self._status = 'new'
		self._worker_id = None
		self._errors = []
		self._starting_time = None
		self._ending_time = None
		self._id = task_id

	def __str__(self):
		return f'{self.id} ({self._status})'

	def __repr__(self):
		return f'{self.__class__.__name__}: {self})'

	@property
	def time_estimate_id(self):
		return 'generic_task'

	@property
	def id(self):
		if self._id is None:
			error = ValueError(f'task id is None!')
			self.add_error(error=error, trace=traceback.format_exc())
			raise error
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

	def get_data_from_namespace(self, namespace, data_id=None, data_slice_id=None):
		return get_data_from_namespace(namespace=namespace, data_id=data_id, data_slice_id=data_slice_id)

	def do(self, namespace, worker_id):
		try:
			self.start()

			# placeholder for whatever needs to be done
			error = NotImplementedError(f'do() is not implemented for this task: {self.id}')
			self.add_error(error=error, trace=traceback.format_exc())
			raise error

			self.end(worker_id=worker_id)
		except Exception as error:
			self.add_error(error=error)

	def end(self, worker_id):
		ending_time = datetime.now()
		if self.status == 'error':
			error = RuntimeError('cannot change error status to done!')
			self.add_error(error=error, trace=traceback.format_exc())
			raise error
		elif self.status == 'started':
			self._status = 'done'
			self._worker_id = worker_id
		else:
			error = RuntimeError(f'unknown status: {self.status}')
			self.add_error(error=error, trace=traceback.format_exc())
			raise error

		self._ending_time = ending_time
		self._worker_id = worker_id

	def get_elapsed(self, unit='ms'):
		if self._starting_time is not None and self._ending_time is not None:
			return get_elapsed(start=self.starting_time, end=self.ending_time, unit=unit)
		else:
			return None

	@property
	def status(self):
		return self._status

	def add_error(self, error, trace=None):
		self._errors.append((error, trace))
		self._status = 'error'

	@property
	def errors(self):
		return self._errors

	def has_error(self):
		return len(self._errors) > 0

	def is_done(self):
		return self._status == 'done'



