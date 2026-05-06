# Este código treina um autoencoder sem regularização
import rede_neural
import leitor_imagem
import pytorch_lightning as L
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import EarlyStopping

# Carregando os dados
imagem, shape, n_pixels = leitor_imagem.read_image("imagem.png")
noised = leitor_imagem.ruido(imagem)
print(torch.cuda.is_available())

imagem = imagem.reshape(1, -1)
noised = noised.reshape(1, -1)
print(imagem.shape == noised.shape)

# Aplicando parada antecipada para evitar sobreajuste

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode='min',
    verbose=False   
)

# Treinando a rede neural:
NUM_EPOCHS = 500
treinador = L.Trainer(callbacks=[early_stopping], max_epochs=NUM_EPOCHS, accelerator="gpu", devices=1)
arquitetura = [n_pixels[0], n_pixels[0] // 2, n_pixels[0] // 4]
autoencoder = rede_neural.Autoencoder(arquitetura, nn.Sigmoid(), nn.MSELoss())
dm = rede_neural.DataModule(imagem, targetData=noised)

treinador.fit(autoencoder, dm)

# Plotando e salvando a curva de aprendizado:
ca_treino = autoencoder.curva_aprendizado_treino
ca_val = autoencoder.curva_aprendizado_val

plt.title("Curva de Aprendizado de treino e validação")
plt.ylabel("Loss")
plt.xlabel("Época")
plt.plot(ca_treino, label="Treino")
plt.plot(ca_val, label="Validação")
plt.legend()
plt.savefig("Autoencoder_incompleto.png")

