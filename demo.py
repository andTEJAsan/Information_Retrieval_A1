from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# Initialize a BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# Set the pre-tokenizer to split on whitespace
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Initialize a trainer for the BPE model
trainer = trainers.BpeTrainer(vocab_size=30000, min_frequency=2, special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"])

# Train the tokenizer on your dataset
files = ["path/to/your/dataset.txt"]
tokenizer.train(files, trainer)

# Save the tokenizer to disk
tokenizer.save("bpe_tokenizer.json")