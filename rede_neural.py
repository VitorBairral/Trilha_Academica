# Este arquivo serve como script padrão para criar as redes neurais a serem utilizadas
#Importações
import numpy as np
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.image import imread
import torch
import torch.nn as nn
import torch.optim as optim

class DataModule(L.LightningDataModule):
    # Classe DataModule, responsável por todo o tratamento dos dados de entrada
    # Importando os dados e transformando em matriz
    def __init__(
            self,
            inputData,
            targetData=None, # Temos um parâmetro de dados de target em caso de uso para denoising 
            n_cores = 1,
            seed = 4002,
            batch_size = 32,
            
    ):
        super().__init__()
        self.n_cores = n_cores
        self.seed = seed
        self.batch_size = batch_size
        self.entrada = inputData
        if targetData == None:
            self.target = self.entrada
        else:
            self.target = targetData
    def setup(self, stage):
        
        # Vamos alimentar cada estágio com a mesma entrada e a mesma saída
        # Estamos trabalhando com aprendizado de máquina não supervisionado comparando a entrada 
        # e a saída, logo, em cada estágio da otimização, os valores permanecerão os mesmos.
        tensor_entrada = torch.tensor(self.entrada, dtype=torch.float32)
        tensor_target = torch.tensor(self.target, dtype=torch.float32)
        if stage == "fit":
            self.input_treino = tensor_entrada
            self.target_treino = tensor_target
            self.input_val = tensor_entrada
            self.target_val = tensor_target

        if stage == "test":
            self.input_test = tensor_entrada
            self.target_test = tensor_target

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.input_treino, self.target_treino),
            batch_size= self.batch_size,
            num_workers=self.n_cores,
            shuffle=False # Não queremos que os dados sejam misturados
        )
    
    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.input_val, self.target_val),
            batch_size= self.batch_size,
            num_workers=self.n_cores,
            shuffle=False # Não queremos que os dados sejam misturados
        )
    
    def test_dataloader(self):
        return DataLoader(
            TensorDataset(self.input_test, self.target_test),
            batch_size= self.batch_size,
            num_workers=self.n_cores,
            shuffle=False # Não queremos que os dados sejam misturados
        )
    
class Autoencoder(L.LightningModule):
    # Por enquanto, faremos um autoencoder inteiro, que contenha o encoder e o decoder.
    def __init__(self, n_input, c_ocultas, fun_ativ, fun_perda):
        super().__init__()

        arquitetura = []
        #Camada de entrada:
        arquitetura.append(nn.Linear(n_input, c_ocultas[0]))
        arquitetura.append(fun_ativ)

        # Para as outras camadas ocultas
        for i in range(1, len(c_ocultas)):
            arquitetura.append(nn.Linear(c_ocultas[i-1], c_ocultas[i]))
            arquitetura.append(fun_ativ)
        
        self.camadas = nn.Sequential(*arquitetura)
        self.fun_perda = fun_perda
        self.perdas_treino = []
        self.perdas_val = []
        self.curva_aprendizado_treino = []
        self.curva_aprendizado_val = []

    def forward(self, x):
        x = self.camadas(x)
        return(x)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    def training_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.fun_perda(y_pred, y)

        self.log("loss", loss, prog_bar=True)
        self.perdas_treino.append(loss)
    def validation_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.fun_perda(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        self.perdas_val.append(loss)
        return loss
    def test_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.fun_perda(y_pred, y)
        self.log("test_loss", loss, prog_bar=True)        
        return loss
    def on_train_epoch_end(self):
        #Atualiza a curva de aprendizado
        perda_media = torch.stack(self.perdas_treino).mean()
        self.curva_aprendizado_treino.append(float(perda_media))
        self.perdas_treino.clear()

    def on_validation_end(self):
        perda_media = torch.stack(self.perdas_val).mean()
        self.curva_aprendizado_val.append(float(perda_media))
        self.perdas_val.clear()
