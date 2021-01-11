import multiprocess
from pandas import DataFrame
from ...time import get_elapsed
from ...time.progress import ProgressBar
from ._worker import worker
from ._Task import Task
from ._TimeEstimate import TimeEstimate


class Manager:
	def __init__(self, evaluation_function, time_unit='ms'):
		self._processes = {}
		self._manager = multiprocess.Manager()

		self._data_namespace = self._manager.Namespace()
		self._data_ids = set()

		self._to_do = self._manager.list()
		self._doing = self._manager.dict()
		self._done = self._manager.list()
		self._processed = []
		self._errors = []
		self._proceed_worker = self._manager.dict()
		self._worker_status = self._manager.dict()
		self._evaluation_function = evaluation_function
		self._time_estimates = {}
		self._tasks_by_id = {}
		self._time_unit = time_unit
		self._worker_id_counter = 0

	def generate_worker_id(self):
		self._worker_id_counter += 1
		return f'worker_{self._worker_id_counter}'

	def add_data(self, data_id, training_data, test_data):
		setattr(self._data_namespace, f'{data_id}_training', training_data)
		setattr(self._data_namespace, f'{data_id}_test', test_data)
		self._data_ids.add(data_id)

	def add_task(self, estimator_id, estimator_class, kwargs, data_id, y_column):
		task = Task(
			estimator_id=estimator_id, estimator_class=estimator_class, kwargs=kwargs,
			data_id=data_id, y_column=y_column
		)
		if task.id in self._tasks_by_id:
			raise ValueError(f'Task {task.id} already exists!')
		self._tasks_by_id[task.id] = task
		self._to_do.append(task)

	def add_task_queue(self, queue):
		"""
		:type queue: list[(str, int, type, dict, DataFrame, DataFrame, str)]
		"""
		for item in queue:
			estimator_id, data_id, estimator_class, kwargs, y_column = item
			self.add_task(
				estimator_id=estimator_id, data_id=data_id, estimator_class=estimator_class,
				kwargs=kwargs, y_column=y_column
			)

	def start(self, num_workers):
		"""
		:type num_workers: int
		"""
		self.process_done_tasks()

		for i in range(num_workers):
			worker_id = self.generate_worker_id()
			process = multiprocess.Process(
				target=worker,
				kwargs={
					'worker_id': worker_id,
					'data_namespace': self._data_namespace,
					'to_do': self._to_do,
					'done': self._done,
					'proceed': self._proceed_worker,
					'status': self._worker_status,
					'evaluation_function': self._evaluation_function
				}
			)
			self._processes[worker_id] = process
			process.start()

		return self._done

	def process_done_tasks(self):
		while True:
			try:
				task = self._done.pop(0)

				if task.status == 'done':
					estimator_type = task.estimator_type
					if estimator_type not in self._time_estimates:
						self._time_estimates[estimator_type] = TimeEstimate()
					self._time_estimates[estimator_type].append(task.get_elapsed(unit=self._time_unit))

				self._processed.append(task)
			except IndexError:
				break

	@property
	def time_estimates(self):
		"""
		:rtype: dict[str, TimeEstimate]
		"""
		return self._time_estimates

	def get_time_estimate(self, task):
		if task.is_done():
			return task.get_elapsed(unit=self._time_unit)

		if len(self.time_estimates) == 0:
			return None

		if task.estimator_type in self._time_estimates:
			return self.time_estimates[task.estimator_type].get_mean()

		mean_estimates = 0
		count = 0
		for estimate in self.time_estimates.values():
			mean_estimates += estimate.get_mean()
			count += 1
		return mean_estimates / count

	def count_to_do(self):
		return len(self._to_do) + len(self._doing)

	def count_done(self):
		return len(self._processed) + len(self._done)

	def get_to_do_time(self):
		if self.count_to_do() == 0:
			return 0

		if len(self.time_estimates) == 0:
			return None

		total = 0
		for task in self._doing.values():
			total += self.get_time_estimate(task=task)

		for task in self._to_do:
			total += self.get_time_estimate(task=task)
		return total

	def get_done_time(self):
		total = 0
		for task in self._processed:
			total += self.get_time_estimate(task=task)
		for task in self._done:
			total += self.get_time_estimate(task=task)
		return total

	def get_num_workers(self):
		active = 0
		idle = 0
		terminated_or_ended = 0
		for worker_status in self._worker_status.values():
			if worker_status in {'started', 'active'}:
				active += 1
			elif worker_status == 'idle':
				idle += 1
			elif worker_status in {'ended', 'terminated'}:
				terminated_or_ended += 1
		result = []
		if active > 0:
			result.append(f'{active} active')
		if idle > 0:
			result.append(f'{idle} idle')
		if terminated_or_ended > 0:
			result.append(f'{terminated_or_ended} ended')

		return ' '.join(result)


	def _update_progress_bar(self, progress_bar):
		self.process_done_tasks()
		to_do_count = self.count_to_do()
		done_count = self.count_done()
		# if there is any time estimate use time to show progress
		if len(self._time_estimates) > 0:
			to_do_time = self.get_to_do_time()
			done_time = self.get_done_time()
			progress_bar.set_total(to_do_time + done_time)
			progress_bar.show(amount=done_time, text=f'{to_do_count} / {to_do_count + done_count} | {self.get_num_workers()}')
		else:
			progress_bar.set_total(to_do_count + done_count)
			progress_bar.show(amount=to_do_count, text=f'{to_do_count} / {to_do_count + done_count} | {self.get_num_workers()}')

		return to_do_count

	def show_progress(self):
		progress_bar = ProgressBar(total=100)
		progress_bar.show(amount=0)
		try:
			while True:
				to_do_count = self._update_progress_bar(progress_bar=progress_bar)
				if to_do_count == 0:
					progress_bar.set_total(total=100)
					progress_bar.show(amount=100)
					break

		except KeyboardInterrupt:
			self._update_progress_bar(progress_bar=progress_bar)

	def worker_status_table(self):
		return DataFrame.from_records([
			{'id': worker_id, 'status': worker_status}
			for worker_id, worker_status in self._worker_status.items()
		])

	def task_table(self):
		d = {}
		for task in self._processed:
			d[task.id] = task
		for task in self._done:
			d[task.id] = task
		for task in self._doing.values:
			d[task.id] = task
		for task in self._to_do:
			d[task.id] = task

		return DataFrame.from_records([
			task.record
			for task in d.values()
		])

	def stop(self, worker_id=None):
		if worker_id is not None:
			self._proceed_worker[worker_id] = False

		else:
			for _worker_id in self._processes.keys():
				self.stop(worker_id=_worker_id)

		return self._done

	def terminate(self, worker_id=None):
		if worker_id is not None:
			self._proceed_worker[worker_id] = False
			self._processes[worker_id].terminate()
			if self._worker_status[worker_id] != 'ended':
				self._worker_status[worker_id] = 'terminated'
		else:
			for _worker_id in self._processes.keys():
				self.terminate(worker_id=_worker_id)
				if self._worker_status[_worker_id] != 'ended':
					self._worker_status[_worker_id] = 'terminated'

		return self._done


