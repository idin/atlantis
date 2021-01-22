from ...collections import cumsum


def split_data(data, count_column, ratios=None, counts=None):
	split_cumsum = data[count_column].cumsum()

	total_sum = data[count_column].sum()

	if counts is None and ratios is not None:
		cumulative_ratios = cumsum(ratios)
		if round(cumulative_ratios[-1], 5) != 1:
			print('ratios', ratios)
			print('total sum', total_sum)
			print('cumulative_ratios', cumulative_ratios)
			raise ValueError('total of ratios is not 1')
		cumulative_ratios[-1] = 1
		cumulative_counts = [x * total_sum for x in cumulative_ratios]

	else:
		cumulative_counts = [round(x) for x in cumsum(counts)]
		if cumulative_counts[-1] != total_sum:
			raise ValueError(f'cumulative count {cumulative_counts[-1]} is not equal total_sum {total_sum} ')

	result = []
	for min_count, max_count in zip([0] + cumulative_counts[:-1], cumulative_counts):
		if min_count == max_count:
			result.append(None)
		else:
			subdata = data[(min_count < split_cumsum) & (split_cumsum <= max_count)]
			result.append(subdata)

	return result
