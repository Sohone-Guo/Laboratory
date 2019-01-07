
import tensorflow as tf
from lib.lstm_attention.layer import LSTM_Attention_Layer
from lib.index_word.util import read_file_sentence_generator
from lib.index_word.data_generator import transorfer_file
import os


if __name__ == "__main__":
    # declare the variable
    tf.app.flags.DEFINE_integer('rnn_cell_unit', 30, 'Number of hidden units in each layer')
    tf.app.flags.DEFINE_integer('num_embedding', 30, 'Embedding dimensions of encoder and decoder inputs')
    tf.app.flags.DEFINE_integer('rnn_layer_num', 20, 'Number of hidden units in each layer')
    tf.app.flags.DEFINE_integer('maximum_iterations', 50, 'Number of hidden units in each layer')
    tf.app.flags.DEFINE_integer('batch_size', 10, 'Batch size')
    tf.app.flags.DEFINE_float('rnn_cell_dropout', 0.5, '')
    tf.app.flags.DEFINE_string('type_models', 'training', 'training or decoder')
    tf.app.flags.DEFINE_string('file_dir', "../../../../Dataset/D/dSummary/Dataset/chinese_news/", '')
    tf.app.flags.DEFINE_string('model_dir', "./data/", '')
    tf.app.flags.DEFINE_string('model_name', "model", '')
    FLAGS = tf.app.flags.FLAGS

    # read the idx and word
    read_word_model = read_file_sentence_generator(file_dir=FLAGS.file_dir,model_dir=FLAGS.model_dir)
    idx_word = read_word_model.index_word
    word_idx = read_word_model.word_index
    length_of_different_words = len(idx_word)

    generator = transorfer_file(read_word_model,FLAGS.file_dir,batch_size=FLAGS.batch_size)
    
    with tf.Session() as sess:
        model = LSTM_Attention_Layer(rnn_cell_unit=FLAGS.rnn_cell_unit,
                                     rnn_cell_dropout=FLAGS.rnn_cell_dropout,
                                     length_of_different_words=length_of_different_words,
                                     num_embedding=FLAGS.num_embedding,
                                     rnn_layer_num=FLAGS.rnn_layer_num,
                                     idx_word=idx_word,
                                     word_idx=word_idx,
                                     maximum_iterations=FLAGS.maximum_iterations,
                                     type_models=FLAGS.type_models)

        sess.run(tf.global_variables_initializer())

        current_step = 0
        for nextBatch in generator:

            # print(nextBatch[0],nextBatch[1],nextBatch[2],nextBatch[3])
            loss, summary,dlt,dt = model.train(sess, nextBatch)

            print("----- Step %d -- Loss %.2f" % (current_step, loss),end="\r",flush=True)
            if current_step %100 == 0 and current_step!=0:
                checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=current_step)

            current_step += 1


