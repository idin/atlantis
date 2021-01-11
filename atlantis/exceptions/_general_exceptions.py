class AtlanteanError(Exception):
	pass


class MissingArgumentError(AtlanteanError, ValueError):
	pass


class AmbiguousArgumentsError(AtlanteanError, ValueError):
	pass


class AmbiguityError(AtlanteanError):
	pass


class MutationError(AtlanteanError):
	pass


class FunctionNotImplementedError(AtlanteanError, AttributeError):
	pass


class ComparisonError(AtlanteanError):
	pass


class EmptyCollectionError(AtlanteanError, ValueError):
	pass
