
import jieba

def sentence_word_list(sentence):
    """[summary]
    
    Arguments:
        sentence {[str]} -- [description]
    """
    list_in_sentence = jieba.cut(sentence)
    return list_in_sentence
