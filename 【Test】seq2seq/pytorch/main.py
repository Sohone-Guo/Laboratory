
from seq2seq.dataset.generator import CustomDataset
from seq2seq.model.seq2seq import Encoder, Decoder
import torch


if __name__ == "__main__":
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    custom_dataset = CustomDataset()
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                            batch_size=1, 
                                            shuffle=True)
    
    embedding_size = 3
    total_word = 22
    hidden_size = 14
    num_layers = 1

    epochs = 100

    
    encoder_model = Encoder(embedding_size, total_word, hidden_size, num_layers)
    decoder_model = Decoder(embedding_size, total_word, hidden_size, num_layers)

    # criterion and optimizer
    loss = torch.nn.CrossEntropyLoss()
    encoder_optimizer = torch.optim.Adadelta(encoder_model.parameters())
    decoder_optimizer = torch.optim.Adadelta(decoder_model.parameters())

    for epoch in range(epochs):
        for idx, (encoder_input, decoder_target) in enumerate(train_loader):

            encoder_input = encoder_input.type(torch.LongTensor).to(device)
            decoder_target = decoder_target.type(torch.LongTensor).to(device)

            # encoder
            encoder_output, encoder_hidden = encoder_model(encoder_input)

            # define loss
            loss_item = 0

            # decoder
            decoder_h = encoder_hidden
            decoder_input = torch.tensor([[0]], dtype=torch.long, device=device)
            for item in range(encoder_input.size(1)):
                decoder_output, decoder_hidden = decoder_model(decoder_input, decoder_h)
                
                inference = torch.max(decoder_output,1)[1].view(1,-1)
                decoder_h = decoder_hidden
                decoder_input = inference

                # print(decoder_output,decoder_target)
                loss_item += loss(decoder_output,decoder_target.view(-1,1)[item])
            
            # loss
            loss_item.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            if (idx) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, epochs, idx+1, len(train_loader), loss_item.item()))

        
    with torch.no_grad():
        correct = 0
        total = 0
        for encoder_input, decoder_target in train_loader:
            encoder_input = encoder_input.type(torch.LongTensor).to(device)
            decoder_target = decoder_target.type(torch.LongTensor).to(device)

            # encoder
            encoder_output, encoder_hidden = encoder_model(encoder_input)

            # define loss
            loss_item = 0

            # decoder
            outputs = []
            decoder_h = encoder_hidden
            decoder_input = torch.tensor([[0]], dtype=torch.long, device=device)
            for item in range(encoder_input.size(1)):
                decoder_output, decoder_hidden = decoder_model(decoder_input, decoder_h)
                
                inference = torch.max(decoder_output,1)[1].view(1,-1)
                decoder_h = decoder_hidden
                decoder_input = inference
                outputs.append(inference[0].data[0])

            outputs = torch.tensor(outputs, dtype=torch.long, device=device)
            print(outputs)

            total += decoder_target.size(1)
            correct += (outputs == decoder_target.view(-1)).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

    # # Save the model checkpoint
    # torch.save(model.state_dict(), 'model.ckpt')