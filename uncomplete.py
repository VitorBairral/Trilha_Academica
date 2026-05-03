# Este código treina um autoencoder sem regularização
import rede_neural
import leitor_imagem
import lightning as L
import torch.nn as nn
import matplotlib.pyplot as plt
import torch

# Carregando os dados
imagem, shape, n_pixels = leitor_imagem.read_image("imagem.png")
print(torch.cuda.is_available())

imagem = imagem.reshape(1, -1)
print(imagem.shape)

# Treinando a rede neural:
NUM_EPOCHS = 500
treinador = L.Trainer(max_epochs=NUM_EPOCHS, accelerator="gpu", devices=1)
arquitetura = [n_pixels[0], n_pixels[0] // 2, n_pixels[0] // 4]
autoencoder = rede_neural.Autoencoder(arquitetura, nn.Sigmoid(), nn.MSELoss())
dm = rede_neural.DataModule(imagem)

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
plt.show()
plt.savefig("Autoencoder_incompleto.png")
