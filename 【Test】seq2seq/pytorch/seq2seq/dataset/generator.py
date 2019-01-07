import torch.utils.data as data 
import torch
import numpy as np

class CustomDataset(data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        pass

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        encoder_input = np.array([1,2,3,4,5,6])
        decoder_target = np.array([6,3,4,3,2,3])
        return encoder_input, decoder_target

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 200


if __name__ == "__main__":

    # You can then use the prebuilt data loader. 
    custom_dataset = CustomDataset()
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                            batch_size=1, 
                                            shuffle=True)
    for item in train_loader:
        print(item)
        break