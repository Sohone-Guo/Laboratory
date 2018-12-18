import torch 
import numpy as np 


class SimpleRNN(torch.nn.Module):

    def __init__(self):
        super(SimpleRNN, self).__init__()

        self.embedding = torch.nn.Embedding(8,3)

        self.rnn = torch.nn.GRU(3,5,2,batch_first=True)

        self.fc = torch.nn.Linear(5,8)
        self.softmax = torch.nn.Softmax()
    
    def forward(self, x):
        embedding = self.embedding(x)

        out, state = self.rnn(embedding)
        
        out = self.fc(out)

        return out


if __name__ == "__main__":
    sentence = {
        "I": 1,
        "want": 2,
        "to": 3,
        "home": 4,
        "go": 5,
        "the": 6,
        "school": 7
    }

    # data = [[1,2,3,4],
    #         [1,5,3,7],
    #         [1,2,5,4]]
    # data = [[0,1,2,3],
    #         [0,2,1,3],
    #         [0,4,1,2],
    #         [0,2,1,4]]
    data = [[1,2,3],
            [2,1,3],
            [4,1,2],
            [2,4,1]]
    
    data = torch.Tensor(data)
    data = data.type(torch.LongTensor)

    model = SimpleRNN()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters())

    for item in range(1000):

        out = model(data)

        loss = 0
        for item in range(out.size(1)):
            loss += criterion(out[:,item,:], data[:,item])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())
    
    with torch.no_grad():
        out = model(data)
        _, predicted = torch.max(out.data, 2)
        print(out)




