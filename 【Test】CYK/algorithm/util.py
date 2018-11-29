"""
This is a test for CYK

-- Grammar

    S -> A B
    A -> C D | C F
    B -> c | E B
    C -> a
    D -> b
    E -> c
    F -> A D

-- Word
    aaabbbcc
"""

from .os import read_content

def read_grammar(path):
    """
    # Arguments:
        path {str}: the address of folder

    # Returns:
        grammar {dic}: {("A","B"):"S", ("a",):"C"}

    """
    grammar = {}

    content = read_content(path)
    content = content.split("\n")
    each_rules = list(map(lambda x:x.strip(),content))
    
    for item_rules in each_rules:
        non_terminal, terminals = item_rules.split("->")
        item_terminals = terminals.split("|")
        item_terminals_strip = list(map(lambda x:x.strip(), item_terminals))
        item_terminals = list(map(lambda x:tuple(x.split(" ")), item_terminals_strip))
        for item_terminal in item_terminals:
            if item_terminal not in grammar:
                grammar[item_terminal] = [non_terminal]
            else:
                grammar[item_terminal].append(non_terminal)
    return grammar


def sentence_to_list(sentence, language):
    """

    # Arguments:
        sentence {str}: "a a a b b b c c"
        language {str}: "english" or "chinese"...

    # Returns:
        list_words {list}: ["a","a","a","b"...]

    """
    return sentence.split(" ")