from joblib import Parallel, delayed
from .BaseController import BaseController
from ._WorkerReport import WorkerReport
from ._Task import Task, Outcome
from ..time.progress import iterate
from ._DEFAULT_VALUES import TIME_UNIT, N_JOBS


class JobController(BaseController):
	def __init__(self, time_unit=TIME_UNIT, n_jobs=N_JOBS):
		super().__init__(time_unit=time_unit)
		self._n_jobs = n_jobs

	def do(self, echo=1):
		worker_id = self._generate_worker_id()
		report = WorkerReport(worker_id=worker_id)

		processor = Parallel(n_jobs=self._n_jobs, backend='threading', require='sharedmem')

		def _do_task(task):
			"""
			:type task: Task
			:rtype: Outcome
			"""
			return task.do(worker_id=worker_id)

		tasks = []
		while self.to_do_count > 0:
			task = self.to_do_queue.popleft()
			self._tasks_status[task.id] = 'started'
			tasks.append(task)

		outcomes = processor(
			delayed(_do_task)(task=task)
			for task in iterate(tasks, echo=echo)
		)

		for outcome in outcomes:
			self.done_queue.append(outcome)
			self._tasks_status[outcome.task_id] = 'done'
			report.add_task_id(task_id=outcome.task_id)

		report.end()
		if echo:
			self.write(f'To-do: {self.to_do_count} - Doing: 0 - Done: {self.done_count} - Processed: 0       ')
		self.process_done_queue(echo=echo)
