def cumsum(l):
	result = []
	total = None
	for x in l:
		if total is None:
			total = x
		else:
			total += x
		result.append(total)
	return result
