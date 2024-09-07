import sys
import multiprocessing
import math
import pickle
import time
# from nltk.tokenize import RegexpTokenizer

import re
from trie import Trie

# In[26]:


special_char = "##"
from collections import defaultdict
import json
delimiters = [' ', ',', '.', ':', ';', '"', '\'', '0', '1', '2','3','4','5','6','7','8','9']
def unique(dump, array):
    if(array):
        dump.append(array[0])
        for i in range(1, len(array)):
            if(array[i] != array[i - 1]): dump.append(array[i])
def simple_tokenize(sentence): # takes a string and returns a list of tokens, removes punctuation whitespaces
	sentence = re.sub(r'[^a-zA-Z0-9 ,.:;"\']+', ' ', sentence)
	pattern = r'[' + re.escape(''.join(delimiters)) + r']+'
	# pattern += r'|[^a-zA-Z0-9\s]'
	# tokz = RegexpTokenizer(pattern, gaps=True)
	# exams me what do you have to do ? 
	tokens = re.split(pattern, sentence.lower())
	tokens = [tok for tok in tokens if tok]
	return tokens
def word_tokenize2(sentence):
	text = sentence
	tokens = []
	i = 0
	n = len(sentence)
	loctok = []
	while(i < n):
		if ord(text[i]) > 127:
			i += 1
			continue	
		if(text[i] in delimiters):
			if(loctok):
				loctok.append('_')
				tokens.append(''.join(loctok))
				loctok = []
			j = i
			while(j < n and text[j] in delimiters):
				j += 1
			i = j
		else: 
			loctok.append(text[i].lower())
			i += 1
	if(loctok):
		tokens.append(''.join(loctok))
	return tokens
	
# def word_tokenize3(sentence):
# 	pattern = r'[' + re.escape(''.join(delimiters)) + r']+'
# 	tokz = RegexpTokenizer(pattern, gaps=True)
# 	# tokens = re.split(pattern, sentence.lower())
# 	tokens = tokz.tokenize(sentence.lower())
# 	for i in range(len(tokens)):
# 		tokens[i] += '_'
# 	return tokens
def word_tokenize(sentence):
	# tokens = tokz.tokenize(sentence.lower(), gaps=True)
	tokens = simple_tokenize(sentence)
	for i in range(len(tokens)):
		tokens[i] += '_'
	return tokens
     
class my_tokenizer:
	def __init__(self, path, sz=10):
		self.tokens = set()
		with open(path, 'r') as f:
			df = [json.loads(line) for line in f]
		# first half of the data
		n = len(df)
		sz = n // 2
		self.df = df[:sz]
		self.merges = []
		self.doc_no_map = {}
		self.doc_id_map = []
		new_df = []
		for index, row in enumerate(self.df):
			if not (row['doc_id'] in self.doc_no_map):
				self.doc_id_map.append(row['doc_id'])
				self.doc_no_map[row['doc_id']] = len(self.doc_no_map) - 1 
				new_df.append(row)
		self.df = new_df
		self.tf  = {}
		self.docf = defaultdict(int)
		self.trie = Trie()
		print(f"len(self.df) = {len(self.df)}")
		print(f"len(self.get_sentences()) = {len(self.get_sentences())}")

		self.inverted_idx = defaultdict(list)
	def get_sentences(self):
		sentences = []
		fields = ['title', 'abstract']
		for index, row in enumerate(self.df):
			for field in fields:
				if field in row:
					sentences.append(str(row[field]))
		return sentences
	def simple_train(self, path):
		sentences = self.get_sentences()
		for sentence in sentences:
			tokens = simple_tokenize(sentence)
			for tok in tokens:
				self.tokens.add(tok)
		self.write_vocabulary_simple(path)
		pass

	def wpe_train(self, k,b, timeout, path):
		t = time.time()
		sentences = self.get_sentences()
		print(f"getting sentences took {time.time() - t}")
		# sentences = ['low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new ']
# 		sentences = [
#     "This is the Hugging Face Course.",
#     "This chapter is about tokenization.",
#     "This section shows several tokenizer algorithms.",
#     "Hopefully, you will be able to understand how they are trained and generate tokens.",
# ]
		t = time.time()
		word_freqs = defaultdict(int)
		for sentence in sentences:
			word_tokens = simple_tokenize(sentence)
			for word in word_tokens:
				word_freqs[word] += 1
		print(f"word freqs took {time.time() - t}")

		# now build the base vocabulary
		# print(word_freqs)
		t = time.time()
		word_splits = {}
		base_vocabulary = self.tokens
		for word in word_freqs.keys():
			initial_split = list(word)
			for i in range(1, len(initial_split)):
				initial_split[i] = special_char + initial_split[i]
				base_vocabulary.add((initial_split[i]))
			if(initial_split):
				base_vocabulary.add((initial_split[0]))
			word_splits[word] = initial_split
		print(f"building base vocabulary in wpe took {time.time() - t}")
		print(f"initial base vocabulary wpe size = {len(base_vocabulary)}")
		merges = []
		t = time.time()
		self.trie = Trie(base_vocabulary)
		print(f"trie building took {time.time() - t}")
		remaining = timeout - (time.time() - t)
		st = time.time()
		while(len(merges) < k):
			if(time.time() - st > remaining): break
			t = time.time()
			pair_freqs = defaultdict(int)
			single_freqs = defaultdict(int)
			pair_words = defaultdict(list)
			for word in word_freqs.keys():
				subword_tok_list = word_splits[word]
				n = len(subword_tok_list)
				for i in range(n):
					single_freqs[subword_tok_list[i]] += word_freqs[word]
				if n == 1: continue
				for i in range(n - 1):
					pair = (subword_tok_list[i], subword_tok_list[i+1])
					pair_freqs[pair] += word_freqs[word]
					pair_words[pair].append(word)
			if(len(pair_freqs) == 0): break
				

			# as an optimization could map the pairs to the words they are in

			max_pair = None
			maxans = - math.inf
			for pair, freq in pair_freqs.items():
				if(single_freqs[pair[0]] * single_freqs[pair[1]] == 0): 
					print(f"1pair = {pair} single_freqs[pair[0]] = {single_freqs[pair[0]]} single_freqs[pair[1]] = {single_freqs[pair[1]]}")
					continue
				# if((2*math.log(freq) - math.log(single_freqs[pair[0]]) - math.log(single_freqs[pair[1]])) > maxans):
				# 	max_pair = pair
				# 	print(f"pair = {pair} single_freqs[pair[0]] = {single_freqs[pair[0]]} single_freqs[pair[1]] = {single_freqs[pair[1]]}")
				# 	maxans = 2 * math.log(freq) - math.log(single_freqs[pair[0]]) - math.log(single_freqs[pair[1]])
				# 	# maxans = (freq * freq) / (single_freqs[pair[0]] * single_freqs[pair[1]])
				# 	print(f"maxans = {maxans}")
				# 	print(f"single_freqs[pair[0]] = {single_freqs[pair[0]]} single_freqs[pair[1]] = {single_freqs[pair[1]]}")
				# 	print(f"pair_freqs[pair] = {pair_freqs[pair]}")
			max_pair = max(pair_freqs, key= lambda x : b*math.log(pair_freqs[x]) - math.log(single_freqs[x[0]]) - math.log(single_freqs[x[1]]))
			# max_pair = max(pair_freqs, key= lambda x : pair_freqs[x] / (single_freqs[x[0]] * single_freqs[x[1]])  )
			if(not max_pair): break
			merged_max_pair = None
			if(max_pair[1].startswith(special_char)):
				merged_max_pair = max_pair[0] + max_pair[1][2:]
			else:
				merged_max_pair = max_pair[0] + max_pair[1]
			base_vocabulary.add(merged_max_pair)
			self.trie.insert(merged_max_pair)
			merges.append((max_pair))
			for word in pair_words[max_pair]:
				if len(word_splits[word]) == 1: continue
				i = 0 
				while i < len(word_splits[word]) - 1:
					subword_tok_list = word_splits[word]
					pair = (subword_tok_list[i], subword_tok_list[i+1])
					if pair == max_pair:
						new_word_splits = subword_tok_list[:i]
						new_word_splits.append(merged_max_pair)
						new_word_splits.extend(subword_tok_list[i+2:])
						word_splits[word] = new_word_splits
					else: i += 1
			print(f"merge {len(merges) - 1} took {time.time() - t} in wpe")
			t = time.time()
			self.write_vocabulary_wpe(path)
			print(f"writing vocabulary took {time.time() - t}")
			# get the most frequent pair
		print(len(merges))
		print(merges)
		print(len(base_vocabulary))
		print(self.tokens)


	def bpe_train(self, k, timeout, path):
		t = time.time()
		sentences = self.get_sentences()
		print(f"getting sentences took {time.time() - t}")
		# sentences = ['low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new ']
		t = time.time()
		word_freqs = defaultdict(int)
		for sentence in sentences:
			word_tokens = word_tokenize(sentence)
			for word in word_tokens:
				word_freqs[word] += 1
		print(f"word freqs took {time.time() - t}")

		# now build the base vocabulary
		# print(word_freqs)
		t = time.time()
		word_splits = {}
		base_vocabulary = self.tokens
		for word in word_freqs.keys():
			for char in word:
				base_vocabulary.add(char)
			word_splits[word] = list(word)
		print(f"building base vocabulary took {time.time() - t}")
		print(f"initial base vocabulary size = {len(base_vocabulary)}")
		merges = []
		remaining = timeout - (time.time() - t)
		st = time.time()
		while(len(merges) < k):
			if(time.time() - st > remaining): break
			t = time.time()
			pair_freqs = defaultdict(int)
			pair_words = defaultdict(list)
			t1 = time.time()
			for word in word_freqs.keys():
				subword_tok_list = word_splits[word]
				n = len(subword_tok_list)
				if n == 1: continue
				for i in range(n - 1):
					pair = (subword_tok_list[i], subword_tok_list[i+1])
					pair_freqs[pair] += word_freqs[word]
					pair_words[pair].append(word)
			print(f"pair freqs took {time.time() - t1}")

			# as an optimization could map the pairs to the words they are in
			# first convert query to t-dimensional space (t = vocab size)
			# convert all documents in the same way
			# use cosine similarity to find the closest document
			# documents are preprocessed ha but saare
			# how ? 

			# max_pair = None
			# max_freq = 0
			# for pair, freq in pair_freqs.items():
			# 	if freq > max_freq:
			# 		max_pair = pair
			# 		max_freq = freq
			if(len(pair_freqs) == 0): break
			max_pair = max(pair_freqs, key=pair_freqs.get)
			if(max_pair == None): break
			base_vocabulary.add(max_pair[0] + max_pair[1])
			merges.append(max_pair)
			t1 = time.time()
			# for word in word_splits.keys():
			for word in pair_words[max_pair]:
				if len(word_splits[word]) == 1: continue
				i = 0 
				while i < (len(word_splits[word]) - 1):
					subword_tok_list = word_splits[word]
					pair = (subword_tok_list[i], subword_tok_list[i+1])
					if pair == max_pair:
						new_word_splits = subword_tok_list[:i]
						new_word_splits.append(max_pair[0] + max_pair[1])
						new_word_splits.extend(subword_tok_list[i+2:])
						word_splits[word] = new_word_splits
					else: i += 1
			print(f"merging took {time.time() - t1}")
			print(f"merge {len(merges) - 1} took {time.time() - t}")
			t = time.time()
			self.write_vocabulary_bpe(path)
			print(f"writing vocabulary took {time.time() - t}")
			# get the most frequent pair
		print(len(merges))
		print(merges)
		print(len(base_vocabulary))
		# self.tokens = base_vocabulary
		self.merges = merges

		# print(word_splits)
		# print(base_vocabulary)
	def write_vocabulary_simple(self, path):
		with open(path, 'w') as f:
			for item in self.tokens:
				f.write(item + ' ')
	def write_vocabulary_bpe(self, path):
		with open(path, 'w') as f:
			for item in self.tokens:
				f.write(item + ' ')
	def write_vocabulary_wpe(self, path):
		with open(path, 'w') as f:
			for item in self.tokens:
				f.write(item + ' ')

	def compute_inverted_index_simple(self):
		for index, row in enumerate(self.df):
			print(f"index = {index}")
			fields = ['title', 'abstract']
			for field in fields:
				tokens = simple_tokenize(str(row[field]))
				for tok in tokens:
					if(len(self.inverted_idx[tok]) == 0):
						self.inverted_idx[tok].append(index)
					elif(self.inverted_idx[tok][-1] != index):
						self.inverted_idx[tok].append(index)

					if tok in self.tf:
						self.tf[tok][index] += 1
					else:
						self.tf[tok] = defaultdict(int)
						self.tf[tok][index] += 1

					self.docf[tok] += (self.tf[tok][index] == 1)
		pass

	def wpe_tokenize(self, words):
		split_words = []
		for word in words:
			split_words.extend(self.trie.greedy_split(word))
		return split_words



	def bpe_tokenize(self, words):
		split_words = []
		for word in words:
			chars = list(word)
			for merge in self.merges:
				i = 0
				while i < len(chars) - 1:
					if (chars[i], chars[i + 1]) == merge:
						new_chars = chars[:i]
						new_chars.append(merge[0] + merge[1])
						new_chars.extend(chars[i+2:])
						chars = new_chars
					else:
						i += 1
			split_words.extend(chars)
		return split_words
    
	def compute_inverted_index_bpe(self):
		t = time.time() 
		for index, row in enumerate(self.df):
			if index % 100 == 0:
				print(f"index = {index} took {time.time() - t}")
				t = time.time()

			fields = ['title', 'abstract']
			for field in fields:
				tokens = word_tokenize(str(row[field]))
				merged = self.bpe_tokenize(tokens)
				for subword_tok in merged:
					if(len(self.inverted_idx[subword_tok]) == 0):
						self.inverted_idx[subword_tok].append(index)
					elif(self.inverted_idx[subword_tok][-1] != index):
						self.inverted_idx[subword_tok].append(index)
					if subword_tok in self.tf:
						self.tf[subword_tok][index] += 1
					else:
						self.tf[subword_tok] = defaultdict(int)
						self.tf[subword_tok][index] += 1	
					self.docf[subword_tok] += (self.tf[subword_tok][index] == 1)
				print(f"index = {index} took {time.time() - t}")
		
	def compute_inverted_index_wpe(self):
		for index, row in enumerate(self.df):
			fields = ['title', 'abstract']
			for field in fields:
		# for index, sentence in enumerate(self.get_sentences()):
				tokens = simple_tokenize(str(row[field]))
				# tokens = simple_tokenize(sentence)
				merged = self.wpe_tokenize(tokens)
				for subword_tok in merged:
					if(len(self.inverted_idx[subword_tok]) == 0):
						self.inverted_idx[subword_tok].append(index)
					elif(self.inverted_idx[subword_tok][-1] != index):
						self.inverted_idx[subword_tok].append(index)
					if subword_tok in self.tf:
						self.tf[subword_tok][index] += 1
					else:
						self.tf[subword_tok] = defaultdict(int)
						self.tf[subword_tok][index] += 1	
					self.docf[subword_tok] += (self.tf[subword_tok][index] == 1)
     
	def write_inverted_index(self, path, opt):
		# want to map from token to a list of docids
		# also compute term frequencies
		options = ["simple", "bpe", "wpe"]
		data = {}
		data["type"] = options[opt]
		data["merges"] = self.merges
		data["length"] = len(self.df)
		data["tf"] = self.tf
		data["docf"] = self.docf
		data["inverted_idx"] = self.inverted_idx
		data["doc_id_map"] = self.doc_id_map
		data["trie"] = self.trie
		with open(path, 'wb') as f:
			pickle.dump(data, f)
		pass



# In[27]:


# import time
# a = my_tokenizer("./train_data/cord19-trec_covid-docs")
# t1 = time.time()
# a.bpe_train(9)
# print(f"time taken = {time.time() - t1}")

# word_tokenize("hello my name is John;doe")


# In[28]:


# a.write_vocabulary("vocab.json")
# t1 = time.time()
# a.bpe_train(18)
# print(f"time taken = {time.time() - t1}")


# In[18]:





# In[22]:


# plt.xlabel('Number of merges')
# plt.ylabel('Time in Seconds')
# plt.xlim(0, max(k_tests) + 1)
# plt.ylim(0, max(times) + 1)
# plt.scatter(k_tests, times)
# plt.savefig('bpe_time.png')


timeout = 295
nsplits = 1000
if __name__ == '__main__':
	train_path = sys.argv[1]
	opt = int(sys.argv[2])
	tokenizer = my_tokenizer(train_path + "/cord19-trec_covid-docs")
	if (opt == 0):
		tokenizer.simple_train('./output.dict')
		tokenizer.write_vocabulary_simple('./output.dict')
	if (opt == 1):
		tokenizer.bpe_train(nsplits,timeout, './output.dict')
		tokenizer.write_vocabulary_bpe('./output.dict')
	if (opt == 2):
		tokenizer.wpe_train(nsplits,2, timeout, './output.dict')
		tokenizer.write_vocabulary_wpe('./output.dict')

     
# In[ ]:





# In[ ]:




