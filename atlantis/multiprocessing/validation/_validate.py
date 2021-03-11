from pandas import DataFrame

from ...ds.validation import ValidationContainer
from ...ds.validation._train_and_test import train_and_test
from .. import ProcessController
from ...time.progress import ProgressBar

from time import sleep


def validate(
		validation_container, controller,
		estimators, evaluation_type, x_columns=None, y_column=None, aggregate=False,
		echo=0
):
	"""
	performs the _evaluate function of ValidationContainer in a multiprocessing manner
	:param validation_container:
	:type  validation_container: ValidationContainer
	:param controller:
	:type  controller: ProcessController
	:param estimators: the estimators to be evaluated
	:type  estimators: list or dict
	:param evaluation_type: regression or classification
	:type  evaluation_type: str
	:type  x_columns: list[str]
	:type  y_column: str
	:param aggregate: if the results should be aggregated per estimator
	:type  echo: int or bool
	:rtype: DataFrame
	"""
	task_ids = set()
	for dictionary in validation_container._generate_evaluation(
		estimators=estimators, evaluation_type=evaluation_type, x_columns=x_columns, y_column=y_column
	):
		estimator = dictionary['kwargs']['estimator']
		try:
			cpu_count = estimator.n_jobs
		except AttributeError:
			cpu_count = 1

		if cpu_count == -1:
			cpu_count = controller.system_cpu_count

		task_id = controller.add_task(
			function=train_and_test,
			kwargs=dictionary['kwargs'],
			cpu_count=cpu_count
		)
		task_ids.add(task_id)

	total_tasks = len(task_ids)
	pb = ProgressBar(total=total_tasks + 1, echo=echo)
	total_processed = 0
	records = []
	while len(task_ids) > 0:
		pb.show(amount=total_processed, text='evaluating folds and estimators in parallel!')
		processed = controller.process_done_queue(echo=0)
		for task_id in processed:
			if task_id in task_ids:
				record = controller.processed[task_id].result
				total_processed += 1
				task_ids.remove(task_id)
				records.append(record)

		sleep(0.5)

	pb.show(amount=total_processed, text=f'aggregating results!')
	df = DataFrame.from_records(records)
	if aggregate:
		df = df.drop(columns='fold').groupby(
			['estimator_id', 'estimator_name']
		).agg(['mean', 'median', 'min', 'max', 'std'])
	pb.show(amount=total_processed + 1, text=f'evaluation complete!')

	return df
