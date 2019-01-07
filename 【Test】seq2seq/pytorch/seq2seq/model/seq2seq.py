import torch 


class Encoder(torch.nn.Module):

    def __init__(self, embedding_size, total_word, hidden_size, num_layers):
        super(Encoder, self).__init__()
        
        self.embedding = torch.nn.Embedding(num_embeddings=total_word,
                                            embedding_dim=embedding_size)
        
        self.gru = torch.nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers)

        self.classify = torch.nn.Linear(hidden_size,total_word)
        

    def forward(self, data):
        data = self.embedding(data)

        # (seq_len, batch, input_size)
        data = data.transpose(0,1)

        out, hidden = self.gru(data)

        # (batch, seq_len, input_size)
        out = out.transpose(0,1)

        return out, hidden


class Decoder(torch.nn.Module):

    def __init__(self, embedding_size, total_word, hidden_size, num_layers):
        super(Decoder, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=total_word,
                                            embedding_dim=embedding_size)

        self.gru = torch.nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers)

        self.classify = torch.nn.Linear(hidden_size,total_word)

    def forward(self, data, hidden):
        data = self.embedding(data)

        out, hidden = self.gru(data, hidden)

        classify = self.classify(out.view(1,-1))

        return classify, hidden

if __name__ == "__main__":
    pass
