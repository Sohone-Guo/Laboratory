
import torch 
from .crf import CRF


class SimpleRNN(CRF):

    def __init__(self, len_words, embedding_dim, hidden_size, num_layers, word_index, START_TAG, STOP_TAG):
        super(SimpleRNN, self).__init__(START_TAG, STOP_TAG, word_index)
        self.START_TAG = START_TAG
        self.STOP_TAG = STOP_TAG
        self.tag_to_ix = word_index
        self.tagset_size = len(self.tag_to_ix)

        self.transitions = torch.nn.Parameter(torch.randn(self.tagset_size,
                                                    self.tagset_size))

        self.transitions.data[self.tag_to_ix[self.START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[self.STOP_TAG]] = -10000


        self.embedding = torch.nn.Embedding(num_embeddings=len_words,
                                            embedding_dim=embedding_dim)
        self.gru = torch.nn.GRU(input_size=embedding_dim,
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                batch_first=True)
        self.liner = torch.nn.Linear(in_features=hidden_size,
                                     out_features=len_words)
    

    def _lstm(self, x):
        embedding = self.embedding(x)

        seq_out, _ = self.gru(embedding)
        seq_out = seq_out.view(seq_out.size(1), -1)

        classification = self.liner(seq_out)
        return classification


    def neg_log_likelihood(self, x, tags):
        feature = self._lstm(x)

        forward_score = self._forward_alg(feature)
        gold_score = self._score_sentence(feature, tags)
        return forward_score - gold_score


    def forward(self, x):  

        feature = self._lstm(x)

        score, tag_seq = self._viterbi_decode(feature)
        return score, tag_seq

