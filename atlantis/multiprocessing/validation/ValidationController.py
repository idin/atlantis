from ..ProcessController import ProcessController
from ._validate import validate


class

class ValidationController(ProcessController):
	def validate(self, validation_container,
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
		return validate(
			validation_container=validation_container, controller=self, estimators=estimators,
			evaluation_type=evaluation_type, x_columns=x_columns, y_column=y_column, aggregate=aggregate,
			echo=echo
		)
