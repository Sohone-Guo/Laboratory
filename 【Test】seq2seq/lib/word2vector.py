from gensim.models import Word2Vec


def word2vector_train(list_sentence, model_name="word2vector.model"):
    """[summary]
    
    Arguments:
        list_sentence {[list]} -- [[["cat", "say", "meow"], ["dog", "say", "woof"]]]
    """
    model = Word2Vec(min_count=1)
    model.build_vocab(list_sentence) 
    model.train(list_sentence, total_examples=model.corpus_count, epochs=model.iter)
    model.save(model_name)
    print("The model trained and saved as {}.".format(model_name))


def word2vector_predict(model_name="word2vector.model"):
    model = Word2Vec.load(model_name)
    return model


if __name__ == "__main__":
    word2vector_train([["cat", "say", "meow"], ["dog", "say", "woof"]])
    model = word2vector_predict()
    print(model.wv["cat"])