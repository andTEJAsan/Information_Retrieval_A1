class TrieNode:
    def __init__(self):
        # Each TrieNode contains a dictionary of children and a boolean flag to indicate the end of a word
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self, words=None):
        # The root node doesn't store any character
        self.root = TrieNode()
        if(words):
            for word in words:
                self.insert(word)
    def longest_prefix(self, word: str) -> str:
        node = self.root
        prefix_ind = -1
        for i, char in enumerate(word):
            if char not in node.children:
                break
            else:
                node = node.children[char]
                if node.is_end_of_word:
                    prefix_ind = i
        return word[:prefix_ind+1]
    def greedy_split(self, word) -> list:
        splits = []

    def insert(self, word: str) -> None:
        # Insert a word into the Trie
        node = self.root
        for char in word:
            # If the character is not present in children, create a new TrieNode
            if char not in node.children:
                node.children[char] = TrieNode()
            # Move to the next node
            node = node.children[char]
        # Mark the end of the word
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        # Search for a word in the Trie
        node = self.root
        for char in word:
            # If character is not found, the word does not exist in the Trie
            if char not in node.children:
                return False
            # Move to the next node
            node = node.children[char]
        # Return True only if the current node marks the end of the word
        return node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        # Check if there is any word in the Trie that starts with the given prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Example usage:
if __name__ == "__main__":
    words = set(["hi", "hello", "apple", "app", "apricot", "banana"])
    trie = Trie(words)
    print(trie.longest_prefix("hiae"))

























