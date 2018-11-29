
import tensorflow as tf
import numpy as np
import jieba
import os

from multiprocessing import Process
import time

from gensim.models import Word2Vec


class word_index_generator(object):
    """
    
    """
    def __init__(self,file_dir,model_dir):
        """ Read txt file
        
        - Arguments
            - file: str
                The address
        
        """
        raw_data = ""
        for item_file_dir in os.listdir(file_dir):
            with open(file_dir+item_file_dir,"rb") as r:
                item_data = r.read()
            try:
                item_data = item_data.decode("utf-8")
                raw_data += item_data + " "
            except:
                item_data = item_data.decode("gbk","ignore")
                raw_data += item_data + " "

        self.raw_data = raw_data
        self.model_dir = model_dir
    

    def save_word_index(self, word,  word_index, index_word):
        """ Save the data to file
        """

        with open(self.model_dir+"word.word","w",encoding="utf-8") as w:
            w.write("{}".format(word))
        
        with open(self.model_dir+"word_index.word","w",encoding="utf-8") as w:
            w.write("{}".format(word_index))    

        with open(self.model_dir+"index_word.word","w",encoding="utf-8") as w:
            w.write("{}".format(index_word))       

        print("Saved: {}, {}, {}, Finished.".format(self.model_dir+"word.word",self.model_dir+"word_index.word",self.model_dir+"index_word.word"))


    def word_index(self):
        """  extracted the word and index

        """
        word = list(set(jieba.cut(self.raw_data)))
        word.append("<go>")
        word.append("<eos>")

        word_index = {word:idx for idx,word in enumerate(word)}
        index_word = {idx:word for idx,word in enumerate(word)}

        self.save_word_index(word,word_index,index_word)


class read_file_sentence_generator(object):
    """
    """
    def __init__(self, file_dir, model_dir):
            # generate the word index
        if not os.path.isfile(model_dir+"word.word"):
            assert model_dir!=None,"The dataset address is None, Please add the file_dir" 
            word_index_generator(file_dir,model_dir).word_index()
        
        self._load_word_index(model_dir)

    def _load_word_index(self,path):
        """
        """
        print("Loading the word index.",end="\r",flush=True)
        with open(path+"word.word","r",encoding="utf-8") as r:
            self.word = eval(r.read())
        
        with open(path+"word_index.word","r",encoding="utf-8") as r:
            self.word_index = eval(r.read())

        with open(path+"index_word.word","r",encoding="utf-8") as r:
            self.index_word = eval(r.read())

        print("Finished the Loading", end="\r",flush=True)

    
    def _sentence_transfer(self,sentence,withGoEos):
        """
        """
        if withGoEos == True:
            sentence_trans = []
            sentence_trans.append(self.word_index["<go>"])
            for word in jieba.cut(sentence):
                if word in self.word_index:
                    sentence_trans.append(self.word_index[word])
            sentence_trans.append(self.word_index["<eos>"])
            return sentence_trans
        else:
            sentence_trans = []
            for word in jieba.cut(sentence):
                if word in self.word_index:
                    sentence_trans.append(self.word_index[word])
            return sentence_trans
    

if __name__ == "__main__":
    read_word_model = read_file_sentence_generator(file_dir="./dataset/ssa.out",model_dir="./data/")

    print(read_word_model)


