import os
import sys
import json
import re
import nltk



# class Vocabulary:

delimiters = [' ', ',', '.', ':', ';', '"', '\'']
def simple_tokenize(text): # takes a string and returns a list of tokens, removes punctuation whitespaces
	tokens = []
	i = 0
	n = len(text)
	loctok = []
	while(i < n):
		if(text[i] in delimiters):
			if(loctok):
				tokens.append(''.join(loctok))
				loctok = []
			j = i
			while(j < n and text[j] in delimiters):
				j += 1
			i = j
		else: 
			loctok.append(text[i])
			i += 1
	if(loctok):
		tokens.append(''.join(loctok))
	return tokens

def basic():
	print(opt)
	pass
def bpe():
	print(opt)
	pass
def wordpiece():
	print(opt)
	pass

class my_tokenizer:
	def __init__(self):
		self.tokens = {}
		pass
	def simple_train(self, path_to_data):
		# read file, contains json objects separated by newlines
		with open(path_to_data, )	


		pass

tokenizers = [basic, bpe, wordpiece]
def main():
	tokenizers[opt]()

if __name__ == '__main__':
	train_path = sys.argv[1]
	opt = int(sys.argv[2])
	main()