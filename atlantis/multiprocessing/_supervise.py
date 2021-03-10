from time import sleep
import multiprocess
from ._do_task import do_task


def superwise(
		to_do_queue, done_queue, processed,
		workers, workers_doing, worker_reports, incomplete_task_ids,
		cpu_usage, max_cpu_count,
		worker_id_counter,
		sleep_time=0.1, echo=0
):
	"""
	:type to_do_queue: multiprocessing.Queue[Task]
	:type done_queue: multiprocessing.Queue[TaskResult]
	:type processed: list[TaskResult]
	:type workers: dict[str, multiprocessing.Process]
	:type workers_doing: dict
	:type worker_reports: dict
	:type incomplete_task_ids: set
	:type cpu_usage: multiprocessing.Value
	:type max_cpu_count: int
	:type worker_id_counter: int
	:type sleep_time: float
	:type echo: bool or int
	"""
	while True:

		# process done queue
		while True:
			try:
				result = done_queue.get_nowait()
			except:
				break
			else:
				processed.append(result)

		# remove dead workers
		dead_worker_ids = set()
		for worker_id, worker in workers.items():
			if not worker.is_alive():
				dead_worker_ids.add(worker_id)
				if worker in workers_doing:
					incomplete_task_id = workers_doing[worker_id]
					if incomplete_task_id is not None:
						incomplete_task_ids.add(incomplete_task_id)
		for dead_worker_id in dead_worker_ids:
			del worker_id[dead_worker_id]

		# add workers as needed
		while not to_do_queue.empty() and cpu_usage.value < max_cpu_count:

			# add worker
			new_worker_id = worker_id_counter.value
			worker_id_counter.value += 1
			process = multiprocess.Process(
				target=do_task,
				kwargs={
					'to_do_queue': to_do_queue,
					'done_queue': done_queue,
					'workers_doing': workers_doing,
					'worker_reports': worker_reports,
					'worker_id': new_worker_id,
					'cpu_usage': cpu_usage,
					'max_cpu_count': max_cpu_count,
					'sleep_time': sleep_time,
					'echo': echo
				}
			)
			workers[new_worker_id] = process
			process.start()