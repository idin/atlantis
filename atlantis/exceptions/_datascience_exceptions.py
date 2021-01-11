from ._general_exceptions import AtlanteanError, AmbiguityError


class DuplicationError(AtlanteanError):
	pass


class DataError(AtlanteanError):
	pass


class MissingColumnsError(DataError):
	pass


class MissingRowsError(DataError):
	pass


class DataAmbiguityError(DataError, AmbiguityError):
	pass


class LearningError(AtlanteanError):
	pass


class ClusteringError(LearningError):
	pass


class ModelNotFittedError(LearningError, RuntimeError):
	pass


class ModelAlreadyFittedError(LearningError, RuntimeError):
	pass


class EstimatorNotEvaluatedError(LearningError, RuntimeError):
	pass


class EvaluationError(LearningError, RuntimeError):
	pass
