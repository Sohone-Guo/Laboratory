
from algorithm.util import read_grammar
from algorithm.cyk import CYK_Algorithm

if __name__ == "__main__":
    sentence = "a a a b b b c c"
    grammar = read_grammar("./configure/grammar.txt")
    parsing = CYK_Algorithm(grammar).parsing_tree(sentence)
    print(parsing)