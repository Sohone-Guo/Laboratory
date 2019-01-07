import numpy as np
import tensorflow as tf 
from tensorflow.python.layers.core import Dense

class LSTM_Attention_Layer(object):
    
    def __init__(self, rnn_cell_unit, rnn_cell_dropout,length_of_different_words,
                    num_embedding,rnn_layer_num, idx_word, word_idx, 
                    maximum_iterations, type_models):
        """[summary]
        
        Arguments:
            rnn_cell_unit {[int]} -- [3]
            rnn_cell_dropout {[float]} -- [0.5]
            length_of_different_words {[int]} -- [3000]
            num_embedding {[int]} -- [30]
            rnn_layer_num {[int]} -- [10]
            idx_word {[dic]} -- [description]
            word_idx {[idc]} -- [description]
            maximum_iterations {[int]} -- [the max length of output]
        """
        self.rnn_cell_unit = rnn_cell_unit
        self.rnn_cell_dropout = rnn_cell_dropout
        self.length_of_different_words = length_of_different_words
        self.num_embedding = num_embedding
        self.rnn_layer_num = rnn_layer_num
        self.idx_word = idx_word
        self.word_idx = word_idx
        self.maximum_iterations = maximum_iterations

        self.network = {}
        self.buildModel(type_models)


    def createRNNCell(self, layer_num):
        def singleRNNCell():
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_cell_unit)
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, 
                                output_keep_prob=self.rnn_cell_dropout)
            return cell
        cell = tf.contrib.rnn.MultiRNNCell([singleRNNCell() for _ in range(layer_num)])
        return cell


    def buildModel(self, type_model="training"):
        """[summary]
        
        Keyword Arguments:
            type_model {str} -- [description] (default: {"training","decoder"})
        """
        # Declare those variable
        self.network["encoder_input"] = tf.placeholder(dtype=tf.int32,
                                                    shape=[None,None],
                                                    name="encoder_input")
        self.network["length_of_encoder_input"] = tf.placeholder(dtype=tf.int32,
                                                        shape=[None],
                                                        name="length_of_encoder_input")
        self.network["decoder_target"] = tf.placeholder(dtype=tf.int32,
                                                    shape=[None,None],
                                                    name="decoder_target")
        self.network["length_of_decoder_target"] = tf.placeholder(dtype=tf.int32,
                                                                shape=[None],
                                                                name="length_of_decoder_target")
        self.network["batch_size"] = tf.placeholder(dtype=tf.int32,
                                                    shape=[None],
                                                    name="batch_size")
        self.network["length_of_max_target_sequence"] = tf.reduce_max(
                                                            self.network["length_of_decoder_target"],
                                                            name="length_of_max_target_sequence")
        self.network["mask"] = tf.sequence_mask(
                                    lengths=self.network["length_of_decoder_target"],
                                    maxlen=self.network["length_of_max_target_sequence"],
                                    dtype=tf.float32,
                                    name="mask")

        # declare the embedding
        self.network["embedding"] = tf.get_variable(name="embedding",
                                                    shape=[self.length_of_different_words,
                                                            self.num_embedding])
        self.network["embedded_encoder_input"] = tf.nn.embedding_lookup(self.network["embedding"],
                                                                self.network["encoder_input"])
        self.network["embedded_decoder_target"] = tf.nn.embedding_lookup(self.network["embedding"],
                                                                self.network["decoder_target"])

        # Declare the encoder
        self.network["encoder_cell"] = self.createRNNCell(layer_num=self.rnn_layer_num)
        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell=self.network["encoder_cell"],
                                                dtype=tf.float32,
                                                inputs=self.network["embedded_encoder_input"],
                                                sequence_length=self.network["length_of_encoder_input"],
                                                time_major=False)
        self.network["encoder_output"] = encoder_output
        self.network["encoder_state"] = encoder_state

        # declare attention
        self.network["attention_mechanism"] = tf.contrib.seq2seq.BahdanauAttention(
                                                num_units=self.rnn_cell_unit,
                                                memory=self.network["encoder_output"],
                                                memory_sequence_length=self.network["length_of_encoder_input"])
        self.network["decoder_cell"] = self.createRNNCell(layer_num=self.rnn_layer_num)
        self.network["decoder_cell"] = tf.contrib.seq2seq.AttentionWrapper(cell=self.network["decoder_cell"],
                                                    attention_layer_size=self.rnn_layer_num,
                                                    attention_mechanism=self.network["attention_mechanism"])
        self.network["decoder_initial_state"] = self.network["decoder_cell"].zero_state(
                                                                                self.network["batch_size"][0],
                                                                                dtype=tf.float32).clone(
                                                                                    cell_state=self.network["encoder_state"])
        self.network["output_layer"] = Dense(self.length_of_different_words)

        # Declare is training or decoder
        if type_model == "training":
            self.network["transpose_embedded_target"] = tf.transpose(self.network["embedded_decoder_target"],
                                                    perm=[1, 0, 2])

            self.network["helper_training"] = tf.contrib.seq2seq.TrainingHelper(
                                                                inputs=self.network["transpose_embedded_target"],
                                                                time_major=True,
                                                                sequence_length=self.network["length_of_decoder_target"],
                                                                name="Train_helper")
            self.network["training_decoder"] = tf.contrib.seq2seq.BasicDecoder(
                                                            cell=self.network["decoder_cell"],
                                                            output_layer=self.network["output_layer"],
                                                            initial_state=self.network["decoder_initial_state"],
                                                            helper=self.network["helper_training"])
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                                                decoder=self.network["training_decoder"],
                                                maximum_iterations=self.network["length_of_max_target_sequence"],
                                                impute_finished=True)
            self.network["decoder_output"] = decoder_output
            self.network["decoder_logits_train"] = tf.identity(self.network["decoder_output"].rnn_output)
            self.network["decoder_predict_train"] = tf.argmax(self.network["decoder_logits_train"], axis=-1,name='decoder_pred_train')
            self.network["loss"] = tf.contrib.seq2seq.sequence_loss(
                                            logits=self.network["decoder_logits_train"],
                                            targets=self.network["decoder_target"],
                                            weights=self.network["mask"])
            tf.summary.scalar("loss",self.network["loss"])
            self.network["summary_op"] = tf.summary.merge_all()
            self.network["optimizer"] = tf.train.AdamOptimizer().minimize(self.network["loss"])

        elif type_model == "decoder":
            
            start_tokens = tf.ones([self.network["batch_size"][0], ], tf.int32) * self.word_idx['<go>']
            end_token = self.word_idx['<eos>']
            self.network["helper_decoder"] = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                                                        embedding=self.network["embedding"],
                                                        start_tokens=start_tokens,
                                                        end_token=end_token)
            self.network["inference_decoder"] = tf.contrib.seq2seq.BasicDecoder(
                                                            cell=self.network["decoder_cell"],
                                                            output_layer=self.network["output_layer"],
                                                            helper=self.network["helper_decoder"],
                                                            initial_state=self.network["decoder_initial_state"])
            decoder_outputs,_,_= tf.contrib.seq2seq.dynamic_decode(decoder=self.network["inference_decoder"],
                                                                    maximum_iterations=self.maximum_iterations)
            self.network["decoder_predict_decode"] = tf.expand_dims(decoder_outputs.sample_id, -1)

        self.saver = tf.train.Saver(tf.global_variables())


    def train(self, sess, batch):
        feed_dict = {self.network["encoder_input"]: batch[0],
                     self.network["length_of_encoder_input"]: batch[1],
                     self.network["decoder_target"]: batch[2],
                     self.network["length_of_decoder_target"]: batch[3],
                     self.network["batch_size"]: [len(batch[0])]}
        _, loss, summary,dlt,dt = sess.run(
                                        [self.network["optimizer"],
                                        self.network["loss"],
                                        self.network["summary_op"],
                                        self.network["decoder_logits_train"],
                                        self.network["decoder_target"]], 
                                        feed_dict=feed_dict)
        return loss, summary,dlt,dt
                                                                
    def infer(self, sess, batch):
        feed_dict = {self.network["encoder_input"]: batch[0],
                     self.network["length_of_encoder_input"]: batch[1],
                     self.network["batch_size"]: [len(batch[0])]}
        predict = sess.run([self.network["decoder_predict_decode"]], feed_dict=feed_dict)
        return predict
    

if __name__ == "__main__":
    model = LSTM_Attention_Layer(rnn_cell_unit=2,
                                     rnn_cell_dropout=0.5,
                                     length_of_different_words=10,
                                     num_embedding=2,
                                     rnn_layer_num=2,
                                     idx_word={},
                                     word_idx={},
                                     maximum_iterations=20,
                                     type_models="training")

