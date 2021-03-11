from .BaseController import BaseController
from ._WorkerReport import WorkerReport


class SimpleController(BaseController):
	def do(self, echo=1):
		worker_id = self._generate_worker_id()
		report = WorkerReport(worker_id=worker_id)
		while self.to_do_count > 0:
			if echo:
				self.write(f'To-do: {self.to_do_count} - Doing: 1 - Done: {self.done_count} - Processed: 0       ')
			task = self.to_do_queue.popleft()
			self._tasks_status[task.id] = 'started'
			outcome = task.do(worker_id=worker_id)
			self.done_queue.append(outcome)
			self._tasks_status[task.id] = 'done'
			report.add_task_id(task_id=task.id)

		report.end()
		if echo:
			self.write(f'To-do: {self.to_do_count} - Doing: 0 - Done: {self.done_count} - Processed: 0       ')
		self.process_done_queue(echo=echo)