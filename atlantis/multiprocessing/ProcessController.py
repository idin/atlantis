import multiprocess
import atexit
from time import sleep
from .BaseController import BaseController
from ._do_task import do_task
from ._DEFAULT_VALUES import *


class ProcessController(BaseController):
	def __init__(
			self,
			time_unit=TIME_UNIT,
			max_cpu_count=MAX_CPU_COUNT,
			sleep_time=SLEEP_TIME,
			empty_count_limit=EMPTY_COUNT_LIMIT,
			max_sleep_time=MAX_SLEEP_TIME
	):
		super().__init__(time_unit=time_unit)

		self._manager = multiprocess.Manager()

		# in parent class:
		self._to_do_queue = self._manager.Queue()
		self._done_queue = self._manager.Queue()
		self._tasks_status = self._manager.dict()
		self._incomplete_task_ids = self._manager.dict()

		# not in parent class:
		self._workers_doing = self._manager.dict()  # key: worker_id, value: task_id
		self._worker_reports = self._manager.dict()
		self._cpu_usage = self._manager.Value('i', 0)

		self._keep_workers_alive = self._manager.Value('i', 1)

		self._workers = dict()

		self._max_cpu_count = max_cpu_count or self.system_cpu_count
		self._sleep_time = sleep_time

		self._empty_count_limit = empty_count_limit
		self._max_sleep_time = max_sleep_time
		atexit.register(self.terminate)

	@property
	def workers_doing(self):
		"""
		:rtype: dict
		"""
		return self._workers_doing

	@property
	def to_do_count(self):
		return self.to_do_queue.qsize()

	@property
	def done_count(self):
		return self.done_queue.qsize()

	@property
	def being_done_count(self):
		return sum(1 for value in self._workers_doing.values() if value is not None)

	@property
	def worker_reports(self):
		"""
		:rtype: dict[str, WorkerReport]
		"""
		return self._worker_reports

	@property
	def tasks_status(self):
		"""
		:rtype: dict
		"""
		return dict(self._tasks_status)

	@property
	def cpu_usage(self):
		return self._cpu_usage.value

	def _add_task_to_to_do(self, task):
		self.to_do_queue.put(task)
		self._tasks_status[task.id] = 'added'
		self._task_counter += 1

	def keep_workers_alive(self):
		self._keep_workers_alive.value = 1

	def add_task(self, function, args=None, kwargs=None, task_id=None, cpu_count=1):
		self.keep_workers_alive()
		return super().add_task(function=function, args=args, kwargs=kwargs, task_id=task_id, cpu_count=1)

	def let_workers_die(self):
		self._keep_workers_alive.value = 0

	@property
	def workers(self):
		"""
		:rtype: dict[str, multiprocessing.Process]
		"""
		return self._workers

	@property
	def worker_count(self):
		return len(self._workers)

	def _generate_worker_id(self, prefix='worker'):
		return super()._generate_worker_id(prefix=prefix)

	def add_worker(self, echo=0):
		worker_id = self._generate_worker_id()
		if echo:
			print(f'adding worker {worker_id}')
		process = multiprocess.Process(
			target=do_task,
			kwargs={
				'to_do_queue': self._to_do_queue,
				'done_queue': self._done_queue,
				'workers_doing': self._workers_doing,
				'tasks_status': self._tasks_status,
				'worker_reports': self._worker_reports,
				'keep_workers_alive': self._keep_workers_alive,
				'worker_id': worker_id,
				'cpu_usage': self._cpu_usage,
				'max_cpu_count': self._max_cpu_count,
				'sleep_time': self._sleep_time,
				'empty_count_limit': self._empty_count_limit,
				'max_sleep_time': self._max_sleep_time,
				'echo': echo
			}
		)
		self._workers[worker_id] = process
		process.start()

	def _insufficient_workers(self):
		return self.cpu_usage < self._max_cpu_count and self.worker_count < self._max_cpu_count

	def add_workers_as_needed(self, echo=0):
		new_worker_count = 0
		while not self._to_do_queue.empty() and self._insufficient_workers():
			sleep(self._sleep_time)
			self.add_worker(echo=echo)
			new_worker_count += 1
		return new_worker_count

	def remove_dead_workers(self, echo=0):
		dead_count = 0
		live_workers = {}
		for worker_id, worker in self.workers.items():
			if worker.is_alive():
				live_workers[worker_id] = worker
			else:
				if echo:
					print(f'removing dead worker {worker_id}')
				if worker_id in self.workers_doing:
					incomplete_task_id = self.workers_doing[worker_id]
					if incomplete_task_id is not None:
						self.incomplete_task_ids[incomplete_task_id] = 1
				dead_count += 1
		self._workers = live_workers
		return dead_count

	def adjust_workers(self, echo=0):
		dead_count = self.remove_dead_workers()
		if self.to_do_count == 0:
			self.let_workers_die()
			new_worker_count = 0
		else:
			self.keep_workers_alive()
			new_worker_count = self.add_workers_as_needed(echo=echo)
		return {'dead': dead_count, 'new': new_worker_count}

	def terminate(self):
		terminated_count = 0
		for worker_id, worker in self.workers.items():
			worker.terminate()
			terminated_count += 1
		return terminated_count

	def _get_from_done_queue(self):
		return self.done_queue.get_nowait()

	def clean_up(self, echo=0):
		processed = self.process_done_queue(echo=echo)
		d = self.adjust_workers(echo=echo)
		dead_count = d['dead']
		new_count = d['new']
		return {'processed_tasks': len(processed), 'dead_workers': dead_count, 'new_workers': new_count}

	def get_status(self, return_text=True, return_values=False):
		todo = self.to_do_count
		being_done = self.being_done_count
		done = self.done_count
		processed = self.processed_count
		cpu_usage = self.cpu_usage
		worker_count = self.worker_count
		if return_values:
			result = {
				'to_do_count': todo,
				'being_done_count': being_done,
				'done_count': done,
				'processed_count': processed,
				'cpu_usage': cpu_usage,
				'worker_count': worker_count
			}
		else:
			result = None

		if return_text:
			text_list = [
				f'CPU: {cpu_usage}/{self._max_cpu_count}',
				f'Workers: {worker_count}',
				f'To-do: {todo}',
				f'Doing: {being_done}/{todo}',
				f'Done: {processed + done}',
				f'Processed: {processed}/{processed + done}        '
			]
			text = ' - '.join(text_list)
			if result is None:
				result = text
			else:
				result['text'] = text

		return result

	def show_status(self, clean_up=1, delay=1):
		try:
			while True:
				self.write(string=self.get_status(return_text=True, return_values=False))
				if clean_up:
					self.clean_up(echo=0)
				sleep(delay)
		except KeyboardInterrupt:
			self.write(string=self.get_status(return_text=True, return_values=False), flush=True)

	def wait_for_tasks(self, echo=1):
		try:
			while True:
				status = self.get_status(return_text=True, return_values=True)
				if echo:
					self.write(string=status['text'])
				d = self.clean_up(echo=0)
				sleep(self._sleep_time)
				if status['to_do_count'] + status['being_done_count'] == 0:
					if status['processed_count'] >= status['done_count']:
						break

		except KeyboardInterrupt:
			return 0

		self.clean_up(echo=0)
		status = self.get_status(return_text=True, return_values=True)
		if echo:
			self.write(string=status['text'] + '\n')

	def do(self, echo=1):
		self.wait_for_tasks(echo=echo)
		self.process_done_queue(echo=echo)
