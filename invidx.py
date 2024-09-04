import sys

from dict_cons import my_tokenizer






if __name__ == '__main__':
    train_path = sys.argv[1]
    idxfile = sys.argv[2]
    opt = int(sys.argv[3])
    tokenizer = my_tokenizer(train_path + "/cord19-trec_covid-docs")
    if opt == 0:
        tokenizer.simple_train()
        tokenizer.write_vocabulary_simple(f"{idxfile}.dict")
        tokenizer.compute_inverted_index_simple()
        tokenizer.write_inverted_index(f"{idxfile}.idx")



