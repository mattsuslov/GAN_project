import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from Discriminator import Discriminator
from Generator import Generator
import pandas as pd
import numpy as np

class cDCGAN(nn.Module):
    def __init__(self, num_classes, photo_size,
                 optimizer=torch.optim.Adam, 
                       criterion=nn.BCELoss, 
                       latent_size=172,
                       EPOCHS=30,
                       lr=3e-4,
                       device=None,):
        super(cDCGAN, self).__init__()
        self.EPOCHS = EPOCHS
        self.optimizer = optimizer
        self.criterion = criterion
        self.latent_size = latent_size
        self.lr = lr
        self.is_path = False
        self.save = False
        self.device = device if device!=None else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
        self.model = {"discriminator": Discriminator(num_classes, photo_size).to(self.device),
                 "generator": Generator(num_classes, self.latent_size).to(self.device)}
        self.optim = {"discriminator": optimizer(self.model["discriminator"].parameters(), lr=lr),
                 "generator": optimizer(self.model["generator"].parameters(), lr=lr)}
        self.crit = {"discriminator": criterion(),
                     "generator": criterion()}

    
    def fit(self, dataloader):
        self.total_loss = {"discriminator": [],
              "generator": []}
        
        BATCH = dataloader.batch_size
        
        true_label = torch.ones(BATCH).to(self.device)
        false_label = torch.zeros(BATCH).to(self.device)

        for epoch in tqdm(range(self.EPOCHS)):
            epoch_loss = {"discriminator": [],
                            "generator": []}
            for xBatch, features1, features2, features3 in tqdm(dataloader):
                xBatch = xBatch.to(self.device)
                features1 = features1.to(self.device)
                features2 = features2.to(self.device)
                features3 = features3.to(self.device)

                self.model["discriminator"].train()
                self.model["generator"].train()

                self.model["discriminator"].zero_grad()
                self.model["generator"].zero_grad()

                real_pred = self.model["discriminator"](xBatch, features1).view(-1)
                real_loss = self.crit["discriminator"](real_pred, true_label)

                latent1 = torch.randn(BATCH, self.latent_size, 1, 1).to(self.device)
                fake_data = self.model["generator"](latent1, features3)
                fake_pred = self.model["discriminator"](fake_data, features3).view(-1)
                fake_loss = self.crit["discriminator"](fake_pred, false_label)

                # discriminator
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.optim["discriminator"].step()

                # generator
                latent2 = torch.randn(BATCH, self.latent_size, 1, 1).to(self.device)
                fake_data2 = self.model["generator"](latent2, features2)
                fake_pred2 = self.model["discriminator"](fake_data2, features2).view(-1)

                g_loss = self.crit["generator"](fake_pred2, true_label)
                g_loss.backward()
                self.optim["generator"].step()

                # data
                epoch_loss["discriminator"].append(d_loss.item())
                epoch_loss["generator"].append(g_loss.item())

            self.total_loss["discriminator"].append(np.mean(epoch_loss["discriminator"]))
            self.total_loss["generator"].append(np.mean(epoch_loss["generator"]))

            if self.save:
                self.save(epoch)
                
    def save(self, epoch):
        torch.save(self.model["discriminator"].state_dict(), f"{self.path_to_save}/discriminator_{epoch}.pt")
        torch.save(self.model["generator"].state_dict(), f"{self.path_to_save}/generator_{epoch}.pt")
        pd.DataFrame({"discriminator": self.total_loss["discriminator"],
                              "generator": self.total_loss["generator"]}).to_csv(f"{self.path_to_save}/loss.csv")
    
    def load_model(self, path_to_generator, path_to_discriminator):
        self.model["discriminator"].load_state_dict(torch.load(path_to_discriminator))
        self.model["generator"].load_state_dict(torch.load(path_to_generator))
        self.epoch = path_to_discriminator.replace(f"{path_to_generator}/generator_", "").replace(".pt", "")
        self.is_path = True
    
    def saving_model(self, path_to_save):
        self.path_to_save = path_to_save
        self.save = True

    def generate(self, features):
        latent = torch.randn(features.shape[0], self.latent_size, 1, 1).to(self.device)
        return self.model["generator"](latent, torch.tensor(features, dtype=torch.float32).to(self.device))