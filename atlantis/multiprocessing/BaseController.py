from collections import deque
import queue
import multiprocess
from pandas import DataFrame
from sys import stdout

from ._Task import Task, Outcome
from ._DEFAULT_VALUES import TIME_UNIT


class BaseController:
	def __init__(self, time_unit=TIME_UNIT):
		self._time_unit = time_unit
		self._to_do_queue = deque()
		self._done_queue = deque()
		self._tasks_status = dict()
		self._processed = dict()
		self._task_counter = 0
		self._tasks_status = dict()
		self._incomplete_task_ids = dict()
		self._worker_id_counter = 0

	def __repr__(self):
		return f'<{self.__class__.__name__} with to-do:{self._get_to_do_plus()}, done:{self.done_count}>'

	def _get_to_do_plus(self):
		return self.to_do_count + self.being_done_count

	@staticmethod
	def write(string, flush=True):
		"""
		:type string: str
		:type flush: bool
		"""
		stdout.write('\r' + string)
		if flush:
			stdout.flush()

	@property
	def system_cpu_count(self):
		return multiprocess.cpu_count()

	@property
	def to_do_queue(self):
		"""
		:rtype: deque[Task] or multiprocessing.Queue[Task]
		"""
		return self._to_do_queue

	@property
	def being_done_count(self):
		return 0

	@property
	def done_queue(self):
		"""
		:rtype: deque[Outcome] or multiprocessing.Queue[Outcome]
		"""
		return self._done_queue

	def _get_from_done_queue(self):
		"""
		:rtype: Outcome
		"""
		return self.done_queue.popleft()

	@property
	def processed_count(self):
		return len(self.processed)

	@property
	def to_do_count(self):
		return len(self.to_do_queue)

	@property
	def done_count(self):
		return len(self.done_queue)

	@property
	def processed(self):
		"""
		:rtype: dict[str, Outcome] or dict[int, Outcome]
		"""
		return self._processed

	def _create_task(self, function, args=None, kwargs=None, task_id=None, cpu_count=1):
		if task_id is None:
			task_id = self._task_counter + 1
		elif isinstance(task_id, int):
			if task_id <= self._task_counter:
				raise ValueError(f'task_id: {task_id} and it cannot be a number smaller than or equal to task counter')
		elif isinstance(task_id, str):
			if task_id in self._tasks_status:
				raise KeyError(f'task_id: "{task_id}" already exists!')
		else:
			raise TypeError(f'task_id of type {type(task_id)} is not acceptable.')

		if cpu_count == -1:
			cpu_count = self.system_cpu_count
		task = Task(
			function=function, task_id=task_id, args=args, kwargs=kwargs, time_unit=self._time_unit,
			cpu_count=cpu_count
		)
		return task

	def _add_task_to_to_do(self, task):
		self.to_do_queue.append(task)
		self._tasks_status[task.id] = 'added'
		self._task_counter += 1

	def add_task(self, function, args=None, kwargs=None, task_id=None, cpu_count=1):
		task = self._create_task(function=function, args=args, kwargs=kwargs, task_id=task_id, cpu_count=cpu_count)
		self._add_task_to_to_do(task=task)
		return task.id

	def get_full_outcome(self, task_id):
		return self.processed[task_id]

	def get_result(self, task_id):
		return self.get_full_outcome(task_id=task_id).result

	@property
	def incomplete_task_ids(self):
		"""
		tasks that are in worker_doing but the worker stops before they finish
		:rtype: set
		"""
		return set(self._incomplete_task_ids.keys())

	def process_done_queue(self, echo=0):
		count = 0
		processed = set()
		while True:
			try:
				outcome = self._get_from_done_queue()
			except (queue.Empty, IndexError):
				break
			else:
				self._tasks_status[outcome.task_id] = 'processed'
				self.processed[outcome.task_id] = outcome
				processed.add(outcome.task_id)
				count += 1
		if echo:
			print(f'{count} tasks processed')
		return processed

	def get_tasks_timing_summary(self):
		"""
		:rtype: DataFrame
		"""
		self.process_done_queue()
		return DataFrame.from_records([
			result.timestamp_record
			for result in self.processed.values()
		])

	@property
	def task_signature_data(self):
		"""
		:rtype: DataFrame
		"""
		self.process_done_queue()
		return DataFrame.from_records([
			result.signature
			for result in self.processed
		])

	def _generate_worker_id(self, prefix='controller'):
		self._worker_id_counter += 1
		return f'{prefix}_{self._worker_id_counter}'

	def get_worker_reports_summary(self, exclude_empty_reports=True):
		result = DataFrame.from_records([
			report.record
			for report in self.worker_reports.values()
		])
		if result.shape[0] > 0:
			if exclude_empty_reports:
				result = result[result['task_count'] > 0]
			result = result.sort_values('start_time').reset_index(drop=True)
		return result

	def do(self, *args, **kwargs):
		raise NotImplementedError('do is not implemented for BaseController')
