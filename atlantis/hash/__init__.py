import base64
import base32hex
import hashlib
from ..collections import make_hashable


def hash_object(obj, base=64):
	"""
	:type obj: Obj
	:type base: int
	:rtype str
	"""
	if hasattr(obj, '__hash64__') and base == 64:
		return obj.__hash64__()
	elif hasattr(obj, '__hash32__') and base == 32:
		return obj.__hash32__()

	hash_maker = hashlib.sha256()
	hash_maker.update(repr(make_hashable(obj)).encode())
	if base == 64:
		return base64.b64encode(hash_maker.digest()).decode()
	elif base == 32:
		return base32hex.b32encode(hash_maker.digest()).replace('=', '-').replace('----', 'x032')
	else:
		raise ValueError(f'base{base} is unknown!')


def hash_file(path, base=64, block_size=65536):
	hash_maker = hashlib.sha256()
	with open(path, 'rb') as file:
		block = file.read(block_size)
		while len(block) > 0:
			hash_maker.update(block)
			block = file.read(block_size)
	if base == 64:
		return base64.b64encode(hash_maker.digest()).decode()
	elif base == 32:
		return base32hex.b32encode(hash_maker.digest()).replace('=', '-')
	else:
		raise ValueError(f'base{base} is unknown!')