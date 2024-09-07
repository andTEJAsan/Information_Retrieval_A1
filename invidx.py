import sys
import time
from dict_cons import my_tokenizer





timeout1 = 30

if __name__ == '__main__':
    train_path = sys.argv[1]
    idxfile = sys.argv[2]
    opt = int(sys.argv[3])
    tokenizer = my_tokenizer(train_path + "/cord19-trec_covid-docs")
    if opt == 0:
        tokenizer.simple_train(f"{idxfile}.dict")
        # tokenizer.write_vocabulary_simple(f"{idxfile}.dict")
        tokenizer.compute_inverted_index_simple(f"{idxfile}.idx")
        t = time.time()
        tokenizer.write_inverted_index(f"{idxfile}.idx", 0)
        print(f"Time taken to write inverted index: {time.time() - t}")
    elif opt == 1:
        tokenizer.bpe_train(4000,timeout1, f"{idxfile}.dict")
        tokenizer.compute_inverted_index_bpe(f"{idxfile}.idx")
        t = time.time()
        tokenizer.write_inverted_index(f"{idxfile}.idx", 1)
        print(f"Time taken to write inverted index: {time.time() - t}")

    elif opt == 2:
        tokenizer.wpe_train(4000,1, timeout1, f"{idxfile}.dict")
        tokenizer.compute_inverted_index_wpe(f"{idxfile}.idx")
        t = time.time()
        tokenizer.write_inverted_index(f"{idxfile}.idx", 2)
        print(f"Time taken to write inverted index: {time.time() - t}")



