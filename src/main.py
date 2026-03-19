import torch
import os
import traceback

from architecture.model import Model
from architecture.configuration import model_configuration

from training.train import train_model
from training.configuration import trn_configuration

from dataengine.dataset import create_dataloader
from dataengine.preprocess_data import pre_tokenize_data
from dataengine.configuration import data_configuration

import tiktoken
from utils.model_manager import save_model, load_model



def main_train():
    trn_bin = data_configuration.get("trn_bin", "")
    val_bin = data_configuration.get("val_bin", "")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(model_configuration).to(device)
            
        tokenizer = tiktoken.get_encoding("gpt2")
        if not os.path.exists(trn_bin) or not os.path.exists(val_bin):
            pre_tokenize_data(data_configuration, tokenizer)

        #training_loader = create_dataloader(trn_bin, data_configuration, shuffle=True)
        #validation_loader = create_dataloader(val_bin, data_configuration, shuffle=False)
        
        #train_model(model, trn_configuration, training_loader, validation_loader, device=device)
        
        load_model(model=model, filepath="output/bckpt.pth")
        save_model(model=model, filepath="output/model.pth")

    except Exception as e:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main_train()

