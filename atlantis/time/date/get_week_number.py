import datetime


def get_week_number(date):
	"""
	:param 	date: a date for which you want the week number of
	:type 	date: datetime.date
	:rtype: int
	"""
	return date.isocalendar()[1]
