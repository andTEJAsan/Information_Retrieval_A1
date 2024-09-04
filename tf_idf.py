import sys

import json
import pickle

if __name__ == '__main__':
    query_file = sys.argv[1]
    result_file = sys.argv[2]
    idxfile = sys.argv[3]
    dictfile = sys.argv[4]
    dictionary = json.load(open(dictfile))
    inv_idx_data = pickle.load(open(idxfile, 'rb'))
    print(inv_idx_data.keys())
    print(inv_idx_data["doc_id_map"])
    print(inv_idx_data["length"])
    if(dictionary["type"] == "simple"):
        vocab = dictionary["vocabulary"]
        
    
    