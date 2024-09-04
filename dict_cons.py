import sys
import pickle
import time

# In[26]:


from collections import defaultdict
import json
delimiters = [' ', ',', '.', ':', ';', '"', '\'']
def unique(dump, array):
    if(array):
        dump.append(array[0])
        for i in range(1, len(array)):
            if(array[i] != array[i - 1]): dump.append(array[i])
def simple_tokenize(sentence): # takes a string and returns a list of tokens, removes punctuation whitespaces
	text = sentence
	tokens = []
	i = 0
	n = len(text)
	loctok = []
	while(i < n):
		if(ord(text[i]) > 127):
			i += 1
			continue
		if(text[i] in delimiters):
			tokens.append((''.join(loctok)))
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
def word_tokenize(sentence):
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
		# self.doc_no_map = {}
		self.doc_id_map = [0] * len(self.df)
		for index, row in enumerate(self.df):
			# self.doc_no_map[row['doc_id']] = index
			self.doc_id_map[index] = row['doc_id']
		self.tf  = {}
		self.docf = defaultdict(int)

		self.inverted_idx = defaultdict(list)
	def get_sentences(self):
		sentences = []
		fields = ['title', 'abstract']
		for index, row in enumerate(self.df):
			for field in fields:
				if field in row:
					sentences.append(str(row[field]))
		return sentences
	def simple_train(self):
		sentences = self.get_sentences()
		for sentence in sentences:
			tokens = simple_tokenize(sentence)
			for tok in tokens:
				self.tokens.add(tok)
		pass
	def bpe_train(self, k):
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
		base_vocabulary = set()
		for word in word_freqs.keys():
			for char in word:
				base_vocabulary.add(char)
			word_splits[word] = list(word)
		print(f"building base vocabulary took {time.time() - t}")
		print(f"initial base vocabulary size = {len(base_vocabulary)}")
		merges = []
		while(len(merges) < k):
			t = time.time()
			pair_freqs = defaultdict(int)
			for word in word_freqs.keys():
				subword_tok_list = word_splits[word]
				n = len(subword_tok_list)
				if n == 1: continue
				for i in range(n - 1):
					pair = (subword_tok_list[i], subword_tok_list[i+1])
					pair_freqs[pair] += word_freqs[word]

			# as an optimization could map the pairs to the words they are in

			# max_pair = None
			# max_freq = 0
			# for pair, freq in pair_freqs.items():
			# 	if freq > max_freq:
			# 		max_pair = pair
			# 		max_freq = freq
			max_pair = max(pair_freqs, key=pair_freqs.get)
			if(max_pair == None): break
			base_vocabulary.add(max_pair[0] + max_pair[1])
			merges.append(max_pair)
			for word in word_splits.keys():
				if len(word_splits[word]) == 1: continue
				i = 0 
				while i < len(word_splits[word]) - 1:
					subword_tok_list = word_splits[word]
					pair = (subword_tok_list[i], subword_tok_list[i+1])
					if pair == max_pair:
						new_word_splits = subword_tok_list[:i]
						new_word_splits.append(max_pair[0] + max_pair[1])
						new_word_splits.extend(subword_tok_list[i+2:])
						word_splits[word] = new_word_splits
					else: i += 1
			print(f"merge {len(merges) - 1} took {time.time() - t}")
			# get the most frequent pair
		print(len(merges))
		print(merges)
		print(len(base_vocabulary))
		self.tokens = base_vocabulary
		self.merges = merges

		# print(word_splits)
		# print(base_vocabulary)
	def write_vocabulary_simple(self, path):
		vocab = {}
		for item in self.tokens:
			vocab[item] = 1
		output = {}
		output["merges"] = self.merges
		output["vocabulary"] = vocab
		output["type"] = "simple"
		json.dump(output, open(path, 'w'))
	def write_vocabulary_bpe(self, path):
		vocab = {}
		for item in self.tokens:
			vocab[item] = 1
		output = {}
		output["merges"] = self.merges
		output["vocabulary"] = vocab
		output["type"] = "bpe"
		json.dump(output, open(path, 'w'))

	def compute_inverted_index_simple(self):
		for index, row in enumerate(self.df):
			fields = ['title', 'abstract', 'doi', 'date']
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
						self.tf[tok] = [0] * len(self.df)
						self.tf[tok][index] += 1

					self.docf[tok] += (self.tf[tok][index] == 1)
		pass
	def bpe_tokenize(self, word):
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
		return chars
    
	def compute_inverted_index_bpe(self):
		for index, row in enumerate(self.df):
			fields = ['title', 'abstract', 'doi', 'date']
			for field in fields:
				tokens = word_tokenize(str(row[field]))
				for tok in tokens:
					merged = self.bpe_tokenize(tok)
					for subword_tok in merged:
						if(len(self.inverted_idx[subword_tok]) == 0):
							self.inverted_idx[subword_tok].append(index)
						elif(self.inverted_idx[subword_tok][-1] != index):
							self.inverted_idx[subword_tok].append(index)
						self.docf[subword_tok] += (self.tf[subword_tok][index] == 0)
						self.tf[subword_tok][index] += 1
			
		
     
	def write_inverted_index(self, path):
		# want to map from token to a list of docids
		# also compute term frequencies
		data = {}
		data["length"] = len(self.df)
		data["tf"] = self.tf
		data["docf"] = self.docf
		data["inverted_idx"] = self.inverted_idx
		data["doc_id_map"] = self.doc_id_map
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


if __name__ == '__main__':
	train_path = sys.argv[1]
	opt = int(sys.argv[2])
	tokenizer = my_tokenizer(train_path + "/cord19-trec_covid-docs")
	if (opt == 0):
		tokenizer.simple_train()
		tokenizer.write_vocabulary_simple('./output.dict')
	if (opt == 1):
		tokenizer.bpe_train(100)
		tokenizer.write_vocabulary_bpe('./output.dict')
     
# In[ ]:





# In[ ]:




