import datetime
from .add_time import add_days
from ..time import get_elapsed


def get_date_mean(dates):
	"""
	:param 	dates: a list or set or tuple of dates
	:type 	dates: list[datetime.date]
	"""

	if len(dates) == 0:
		raise ValueError(f'dates is empty')

	first_date = dates[0]
	total_extra_days = sum([get_elapsed(start=first_date, end=date, unit='day') for date in dates[1:]])
	mean_extra_days = total_extra_days / len(dates)
	return add_days(first_date, days=round(mean_extra_days))
