# Este código treina um autoencoder sem regularização
import rede_neural
import leitor_imagem
import lightning as L
import torch.nn as nn
import matplotlib.pyplot as plt

# Carregando os dados
imagem, shape, n_pixels = leitor_imagem.read_image("imagem.png")

# Treinando a rede neural:
NUM_EPOCHS = 500
treinador = L.Trainer(max_epochs=NUM_EPOCHS)
c_ocultas = [67500, 50625, 37969, 50625, 67500, 90000]
autoencoder = rede_neural.Autoencoder(n_pixels, c_ocultas, nn.Sigmoid, nn.MSELoss)
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
plt.savefig("Autoencoder_incompleto.png")
