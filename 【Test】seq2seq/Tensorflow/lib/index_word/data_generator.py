
from lib.index_word.util import *
import numpy as np
import random
import jieba
import itertools

import sys 
sys.path.append("C:\项目\Lib")
from NLP_Lib.sentence_token import sentenceTaken


def list_array(data,data_len_T,data_len_I):

    sentences = []
    max_value = max(data_len_T) #max(max(data_len_T),max(data_len_I))
    for sentence in data:
        complemented_value = max_value - len(sentence)
        sentences.append(sentence+[sentence[-1]]*complemented_value)
    return np.array(sentences)#.reshape(max_value,-1)



# def pickedWord(sentence):
#     """[summary]
    
#     Arguments:
#         sentence {[str]} -- [description]
#     """
#     words = list(jieba.cut(sentence))
#     len_words = len(words)
#     start = 2# int(2+len_words/10)
#     end = 4#int(2+(len_words*4)/10)
#     num = random.randint(a=start,b=end)
#     total_combination = list(itertools.combinations(words,num))
#     random.shuffle(total_combination)
#     for item in total_combination:
#         item = list(item)
#         # idx = words.index(item[0])
#         # random.shuffle(item)
#         return (" ".join(item), sentence)


def pickedWord(sentence):
    """[summary]
    
    Arguments:
        sentence {[str]} -- [description]
    """
    words = list(jieba.cut(sentence))
    len_words = len(words)
    start = 2# int(2+len_words/10)
    end = 4#int(2+(len_words*4)/10)
    num = random.randint(a=start,b=end)
    total_combination = list(itertools.combinations(words,num))
    random.shuffle(total_combination)
    for item in total_combination:
        item = list(item)
        idx = words.index(item[0])
        # random.shuffle(item)
        return (" ".join(item), " ".join(words[idx:]))

# def pickedWord(sentence):
#     """[summary]
    
#     Arguments:
#         sentence {[str]} -- [description]
#     """
#     words = list(jieba.cut(sentence))
#     len_words = len(words)
#     start = 0
#     end = len_words-1
#     num = random.randint(a=2,b=3)
#     total_combination = list(itertools.combinations(words,num))
#     random.shuffle(total_combination)
#     for item in total_combination:
#         item = list(item)
#         random.shuffle(item)
#     return (words[start], "".join(words[start:]))



def transorfer_file(word_model,file_dir,batch_size,language="chinese"):
    """[summary]
    
    Arguments:
        word_model {[class]} -- [description]
        file_dir {[dir]} -- [description]
        batch_size {[int]} -- [description]
    
    Keyword Arguments:
        language {str} -- [description] (default: {"chinese"})
    """
    # read sentence from file
    sentences_tranfer_input,sentences_len_input = [],[]
    sentences_tranfer_target,sentences_len_target = [],[]
    list_files_name = os.listdir(file_dir)
    while True:
        random.shuffle(list_files_name)
        for item_file in list_files_name:
            try:
                with open(file_dir+item_file,"rb") as r:
                    try:
                        data = r.read().decode("utf-8")
                    except:
                        continue
                    for sentence in sentenceTaken(text=data,language=language):
                        # print("Sentence is ",sentence)
                        picked_word, target = pickedWord(sentence)
                        picked_word_transfer =  word_model._sentence_transfer(picked_word.strip(),withGoEos=False)
                        target_transfer = word_model._sentence_transfer(target.strip(),withGoEos=True)

                        if len(sentences_tranfer_target) != batch_size:
                            sentences_tranfer_input.append(picked_word_transfer)
                            sentences_len_input.append(len(picked_word_transfer))
                            sentences_tranfer_target.append(target_transfer)
                            sentences_len_target.append(len(target_transfer))

                        if len(sentences_tranfer_target) == batch_size:
                            sentence_transfer_array_input = list_array(sentences_tranfer_input,sentences_len_input,sentences_len_target) 
                            sentence_transfer_array_target = list_array(sentences_tranfer_target,sentences_len_target,sentences_len_input) 
                            yield [sentence_transfer_array_input,np.array(sentences_len_input),sentence_transfer_array_target,np.array(sentences_len_target)]

                            sentences_tranfer_input,sentences_len_input = [],[]
                            sentences_tranfer_target,sentences_len_target = [],[]
            except Exception as e:
                # print(item_file)
                pass
               

def transorfer_file_infer(word_model,words,batch_size,language="chinese"):
    """[summary]
    
    Arguments:
        word_model {[class]} -- [description]
        file_dir {[dir]} -- [description]
        batch_size {[int]} -- [description]
    
    Keyword Arguments:
        language {str} -- [description] (default: {"chinese"})
    """
    # read sentence from file
    sentences_tranfer_input,sentences_len_input = [],[]
    sentences_tranfer_target,sentences_len_target = [],[]


                # print("Sentence is ",sentence)
    # picked_word, target = pickedWord(words)
    picked_word = words
    target = "None"
    picked_word_transfer =  word_model._sentence_transfer(picked_word.strip(),withGoEos=False)
    target_transfer = word_model._sentence_transfer(target.strip(),withGoEos=True)


    sentences_tranfer_input.append(picked_word_transfer)
    sentences_len_input.append(len(picked_word_transfer))
    sentences_tranfer_target.append(target_transfer)
    sentences_len_target.append(len(target_transfer))

    sentence_transfer_array_input = list_array(sentences_tranfer_input,sentences_len_input,sentences_len_target) 
    sentence_transfer_array_target = list_array(sentences_tranfer_target,sentences_len_target,sentences_len_input) 
    return [sentence_transfer_array_input,np.array(sentences_len_input),sentence_transfer_array_target,np.array(sentences_len_target)]


if __name__ == "__main__":
    generator = transorfer_file(read_word_model,"../dataset/corpus/ssa.out",batch_size=1)