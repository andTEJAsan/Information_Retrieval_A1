import time
import sys
import math
import heapq

import json
import pickle
from trie import Trie
from dict_cons import simple_tokenize
from dict_cons import word_tokenize

if __name__ == '__main__':
    query_file = sys.argv[1]
    result_file = sys.argv[2]
    idxfile = sys.argv[3]
    dictfile = sys.argv[4]
    inv_idx_data = pickle.load(open(idxfile, 'rb'))
    vocab = []
    with open(dictfile, 'r') as f:
        for line in f:
            # print(type(line))
            # print(line)
            vocab.extend(line.split())
    print(vocab)
    trie = Trie(vocab)
    print(inv_idx_data.keys())
    if(inv_idx_data["type"] == "simple"):
        print("simple")
    with open(result_file, 'w') as f:
        f.write("qid iteration docid relevancy\n")
    with open(result_file, 'a') as f:
        f.write("qid iteration docid relevancy\n")
    with open(query_file, 'r') as f:
        queries = [json.loads(line) for line in f]

    

    def wpe_tokenize(words):
        split_words = []
        for word in words:
            split_words.extend(trie.greedy_split(word))
        return split_words

    def bpe_tokenize(words):
        split_words = []
        for word in words:
            split_words.extend(trie.greedy_split2(word))
        return split_words
    def vectorize(text):
        # convert text to a list of tokens
        pretoks = None
        tokens = []
        if(inv_idx_data["type"] == "simple"):
            pretoks = simple_tokenize(text)
            tokens = pretoks
        elif(inv_idx_data["type"] == "bpe"):
            pretoks = word_tokenize(text)
            tokens = bpe_tokenize(pretoks)
        elif(inv_idx_data["type"] == "wpe"):
            pretoks = simple_tokenize(text)
            tokens = wpe_tokenize(pretoks)
        return tokens
        
    # query * document / ||document||
    docf = inv_idx_data["docf"]
    n = inv_idx_data["length"]
    tf = inv_idx_data["tf"]
    mags = inv_idx_data["docmags"]
    doc_id_map = inv_idx_data["doc_id_map"]
    # print(mags)
    def answer_query(query):
        # convert (query) to a list of tokens
        qid = query["query_id"]
        title_str = query["title"]
        narrative_str = query["narrative"]
        description_str = query["description"]
        title_vec = vectorize(title_str)
        narrative_vec = vectorize(narrative_str)
        description_vec = vectorize(description_str)

        def dot(query_vec, j):
            # dot product of query_vec and document_vec[j]
            # return the dot product
            prod = 0
            for tok in query_vec:
                prod += math.log(1 + n / docf[tok]) * tf[tok][j] * math.log(1 + n / docf[tok])
            return prod

        mh = []
        a = 2
        b = 3
        c = 1
        for ind in range(inv_idx_data["length"]):
            score = a * dot(title_vec, ind) +  b * dot(narrative_vec, ind) + c * dot(description_vec, ind)
            try:
                score = score / mags[ind]
            except:
                continue
            if(len(mh) == 0):
                heapq.heappush(mh, (score, ind))
            elif(score > mh[0][0]):
                heapq.heappush(mh, (score, ind))
            if(len(mh) > 100):
                heapq.heappop(mh)

        mh.sort(reverse=True)
        for i in range(len(mh)):
            doc_id = doc_id_map[mh[i][1]]
            with open(result_file, 'a') as f:
                f.write(f"{qid} 0 {doc_id} {mh[i][0]}\n")



    i = 0
    t = time.time()
    for query in queries:   
        i += 1
        print(f"i = {i}")
        print(f"Answering query {query['query_id']}")
        answer_query(query)
        print(f"Time taken to answer query: {time.time() - t}")
        t = time.time()
    
    