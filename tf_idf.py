import sys

import json
import pickle

if __name__ == '__main__':
    query_file = sys.argv[1]
    result_file = sys.argv[2]
    idxfile = sys.argv[3]
    dictfile = sys.argv[4]
    inv_idx_data = pickle.load(open(idxfile, 'rb'))
    vocab = 
    with open(dictfile, 'r') as f:
        for line in f:

    print(inv_idx_data.keys())
    print(inv_idx_data["doc_id_map"])
    print(inv_idx_data["length"])
    if(inv_idx_data["type"] == "simple"):
        vocab = ["vocabulary"]
        
    
    