
from .util import sentence_to_list
from .tree import BTree
import numpy as np


class CYK_Algorithm():
    """ Read the gammar and 
    return a sentence tree

    # Arguments:
        grammar {dic}: {("A","B"):"S", ("a",):"C"}

    # Returns:
    
    """
    def __init__(self, grammar, language="chinese"):
        self.grammar = grammar
        self.language = language

    def parsing_tree(self, sentence):
        """
        # Arguments:
            sentence {str}: "a a a b b b c c"

        """
        matrix_dic = {}

        list_words = sentence_to_list(sentence,self.language)
        len_words = len(list_words)

        # make the first
        for index_hor in range(len_words):
            word = list_words[index_hor]
            label = self.grammar[(word,)]
            matrix_dic[(index_hor,0)] = {word:[]}
            for label_item in label:
                matrix_dic[(index_hor,0)][word].append(label_item)
  
        for idx_vec in range(1,len_words):
            for index_hor in range(len_words-idx_vec):
                matrix_dic[(index_hor,idx_vec)] = {}
                position = index_hor
                for index in range(idx_vec):
                    position += 1
                    index_a = (index_hor,index)
                    index_b = (position, idx_vec-index-1)

                    if index_a in matrix_dic and index_b in matrix_dic:
                        result = list(self.merge_label(matrix_dic, index_a, index_b))
                        if result != []:
                            matrix_dic[(index_hor,idx_vec)][(index_a,index_b)] = result

                if matrix_dic[(index_hor,idx_vec)] == {}:
                    del matrix_dic[(index_hor,idx_vec)]

        return BTree(matrix_dic).build_tree()
                        

    def merge_label(self, matrix_dic, index_a, index_b):
        """
        # Arguments
            matrix_dic {list}
            index_a {dic}
            index_b {dic}
        
        # Returns
            yield  ['A ']

        """
        # read the dictionary
        dic_a = matrix_dic[index_a]
        dic_b = matrix_dic[index_b]
        # read the value
        value_a = list(map(lambda x:dic_a[x], dic_a))[0]
        value_b = list(map(lambda x:dic_b[x], dic_b))[0]
        # strip
        value_a = list(map(lambda x:x.strip(), value_a))
        value_b = list(map(lambda x:x.strip(), value_b))
        for item_a in value_a:
            for item_b in value_b:
                if (item_a,item_b) in self.grammar:
                    yield self.grammar[(item_a,item_b)][0]
        



