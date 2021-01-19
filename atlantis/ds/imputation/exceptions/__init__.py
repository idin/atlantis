class ImputationError(Exception):
	pass


class ColumnError(ImputationError):
	pass


class NoColumnsError(ColumnError):
	pass


class MissingColumnError(ColumnError):
	pass


class ColumnTypeError(ColumnError):
	pass


class InputError(ImputationError):
	pass


class IncludeExcludeClashError(ColumnError):
	pass


class ImputerNotFittedError(ImputationError):
	pass


class MissingModelError(ImputationError):
	pass
