from .SimpleController import SimpleController
from .JobController import JobController
from .ProcessController import ProcessController
from ._DEFAULT_VALUES import *


def Controller(
	model,
	time_unit=TIME_UNIT,
	max_cpu_count=MAX_CPU_COUNT,
	sleep_time=SLEEP_TIME,
	empty_count_limit=EMPTY_COUNT_LIMIT,
	max_sleep_time=MAX_SLEEP_TIME,
	n_jobs=N_JOBS
):
	if not isinstance(model, str):
		raise ValueError('controller_type should be a string')
	model = model.lower()

	if model in ['s', 'simple controller', 'simple']:
		return SimpleController(time_unit=time_unit)
	elif model in ['j', 'job controller', 'job', 'jobs']:
		return JobController(time_unit=time_unit, n_jobs=n_jobs)
	elif model in ['p', 'm', 'process controller', 'multiprocess controller', 'process', 'multiprocess']:
		return ProcessController(
			time_unit=time_unit, max_cpu_count=max_cpu_count,
			sleep_time=sleep_time, empty_count_limit=empty_count_limit, max_sleep_time=max_sleep_time
		)
	else:
		raise ValueError(f'unknown controller type: "{model}"')
