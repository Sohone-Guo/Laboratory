
import os 

import torch

from property.util.os import read_content
from property.dataset.generator import PropertyDataset

from property.model.property import Model
from property.trainer.training import training_model
from property.evaluator.eval import val_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,6,7,8,9"

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configure = eval(read_content("./configure"))


    # train dataset
    property_dataset = PropertyDataset(configure, status="train")
    property_loader = torch.utils.data.DataLoader(dataset=property_dataset,
                                                  batch_size=configure["batch_size"],
                                                  shuffle=True)
                                            
    # val dataset
    property_dataset_val = PropertyDataset(configure, status="val")
    property_loader_val = torch.utils.data.DataLoader(dataset=property_dataset_val,
                                                  batch_size=configure["val_batch_size"],
                                                  shuffle=False)

    model = Model().to(device)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])

    # call the training
    for epoch in range(configure["epoch_size"]):
        model = training_model(model, property_loader, configure, device, epoch)
        val_model(model, property_loader_val, configure, device)

    
