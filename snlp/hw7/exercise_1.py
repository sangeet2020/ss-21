from collections import defaultdict
import math
import pdb


class TrieNode(object):
    def __init__(self, char):
        self.char = char
        self.is_end = False
        self.counter = 0
        self.children = {}

class CountTree():
    def __init__(self, n=4):
        self.n = n
        self.root = TrieNode("")
        self.ngram_tree = defaultdict()
        
    def add(self, ngram):
        node = self.root
        ngram = self.reverse(ngram)
        
        for char in ngram:
            if char in node.children:
                node = node.children[char]
            else:
                # If a character is not found,
                # create a new node in the trie
                new_node = TrieNode(char)
                node.children[char] = new_node
                node = new_node
        
        # Mark the end of a word
        node.is_end = True
        node.counter += 1
        
    def dfs(self, node, prefix):
        """Depth-first traversal of the trie
        """
        # pdb.set_trace()
        if node.is_end:
            self.output.append((prefix + node.char, node.counter))
        
        for child in node.children.values():
            self.dfs(child, prefix + node.char)


    def get(self, ngram):
        self.output = []
        node = self.root
        ngram = self.reverse(ngram)
        # Check if the prefix is in the trie
        # pdb.set_trace()
        for char in ngram:
            if char in node.children:
                node = node.children[char]
            else:
                # cannot found the prefix, return empty list
                return 0
        
        # Traverse the trie to get all candidates
        self.dfs(node, ngram)
        sorted_list = sorted(self.output, key=lambda x: x[1], reverse=True)
        # pdb.set_trace()
        counts = 0
        for pair in sorted_list:
            counts += pair[1]
        return counts


    def perplexity(self, ngrams, vocab):
        pass
    
    def prune(self, k):
        pass
    
    def reverse(self, s):
        s1 = ''.join(reversed(s))
        return s1
    
    def PrintTree(self):
        # pdb.set_trace()
        for i in self.root.children.values():
            print(i, i.children)
				

# t = CountTree()
# t.add("ABCE")
# t.add("ABCD")
# t.add("ABCD")
# t.add("QBCD")
# t.add("QQCD")
# t.add("BCDA")

# t.add("1234")
# t.add("1234")
# t.add("1234")
# t.add("1234")
# t.add("1234")
# t.add("5634")

# # t.PrintTree()

# print(t.get("ABCD"))
# print(t.get("ABCX"))
# print(t.get("BCD"))
# print(t.get("D"))
# print(t.get("CD"))
# print(t.get("1234"))
# print(t.get("5634"))