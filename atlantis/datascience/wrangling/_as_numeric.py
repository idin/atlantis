from numpy import where, nan, int64
from pandas import to_numeric

def as_int(x):
	return where(x.astype(str).str.isnumeric(), x.astype(str).astype(int64, errors='ignore'), nan)


def as_numeric(x):
	return to_numeric(x, errors='coerce')