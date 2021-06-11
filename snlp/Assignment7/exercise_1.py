
from collections import defaultdict
import math

class CountTree():
	def __init__(self, n=4):
		self.root = dict()

	def add(self, ngram):
		curr_dict = self.root	
		# for c in ngram[::-1]: # iterating in reverse order
		# 	if c in curr_dict:
		# 		curr_dict = curr_dict[c]
		# 	else: # initialize new node
		# 		curr_dict[c] = None
		self.root = self.recur_add(self.root, ngram)
	
	
	def recur_add(self, subtree, sub_ngram):
		if sub_ngram[-1] in subtree: # if found in subtree
			if len(sub_ngram) == 1:
				# subtree[sub_ngram[-1]]
				subtree['cnt'] += 1 # increment count
				return subtree

			if sub_ngram[-2] not in subtree[sub_ngram[-1]]: # new branch
				new_subtree = self.recur_add({}, sub_ngram[:-1])
				# subtree['cnt'] = subtree[sub_ngram[-1]]['cnt'] + new_subtree['cnt'] # summing count
				subtree[sub_ngram[-1]] = {**subtree[sub_ngram[-1]], **new_subtree}	# branching out
			else:
				# continue traversing in tree
				subtree[sub_ngram[-1]] = self.recur_add(subtree[sub_ngram[-1]], sub_ngram[:-1])
				subtree['cnt'] = subtree[sub_ngram[-1]]['cnt']
			return subtree

		else: # if new node
			if len(sub_ngram) == 1: # at leaf node
				subtree[sub_ngram] = None # leaf node
				subtree['cnt'] = 1 # add new count
				return subtree
			
			# recursively adding new nodes
			new_subtree = self.recur_add({}, sub_ngram[:-1])
			# in between tree
			subtree[sub_ngram[-1]] = new_subtree
			subtree['cnt'] = new_subtree['cnt'] # count at each node

			return subtree

	def get(self, ngram):
		if len(ngram) == 0:
			return 0
		cnt = self.recur_get(self.root, ngram)
		return cnt

	def recur_get(self, subtree, sub_ngram):
		if sub_ngram[-1] in subtree:
			if len(sub_ngram) == 1:
				# if type(subtree[sub_ngram]) == type(1):
				# 	return subtree[sub_ngram]
				# else:
				# 	return subtree['cnt']
				return subtree['cnt']
			return self.recur_get(subtree[sub_ngram[-1]], sub_ngram[:-1])
		else:
			return 0

	def perplexity(self, ngrams, vocab):
		pass

	def prune(self, k):
		pass

	