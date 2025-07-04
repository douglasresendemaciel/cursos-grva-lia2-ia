# treinar_e_visualizar_iris.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Importa a classe Perceptron do seu arquivo perceptron.py
# Este script assume que o arquivo 'perceptron.py' está no mesmo diretório.
from perceptron import Perceptron

# ---------------------------------------------------------------------------
# FUNÇÃO AUXILIAR PARA VISUALIZAÇÃO DA FRONTEIRA DE DECISÃO
# (Esta função permanece a mesma)
# ---------------------------------------------------------------------------
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=f'Classe {cl}', edgecolor='black')
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='none', edgecolor='black',
                    alpha=1.0, linewidth=1, marker='o', s=100, label='Amostras de Teste')

# ---------------------------------------------------------------------------
# SCRIPT PRINCIPAL DE EXECUÇÃO
# ---------------------------------------------------------------------------

# --- Passo 1: Carregar e preparar os dados ---
# https://www.kaggle.com/datasets/uciml/iris/data
iris = load_iris()
X = iris.data[:, [0, 2]]
y = np.where(iris.target == 0, 1, 0) # 1 para Setosa, 0 para outras

# --- NOVO: Passo de Visualização dos Dados de Entrada (EDA) ---
print("Exibindo a visualização dos dados de entrada...")
plt.figure(figsize=(8, 6))
# Plota as amostras da Classe 1 (Setosa)
plt.scatter(X[y == 1, 0], X[y == 1, 1],
            color='red', marker='o', label='Setosa (Classe 1)')
# Plota as amostras da Classe 0 (Não Setosa)
plt.scatter(X[y == 0, 0], X[y == 0, 1],
            color='blue', marker='s', label='Não Setosa (Classe 0)')

plt.xlabel('Comprimento da Sépala [cm]')
plt.ylabel('Comprimento da Pétala [cm]')
plt.title('Visualização dos Dados de Entrada - Dataset Iris')
plt.legend(loc='upper left')
plt.grid(True)
plt.show() # Exibe o primeiro gráfico. O script pausa aqui até a janela ser fechada.


# --- Passo 2: Dividir os dados em treino e teste ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)
print(f"\nTotal de amostras: {len(y)}")
print(f"Amostras de treino: {len(y_train)}")
print(f"Amostras de teste: {len(y_test)}")

# --- Passo 3: Treinar o Perceptron ---
print("\nTreinando o Perceptron...")
ppn = Perceptron(learning_rate=0.1, n_epochs=10)
ppn.fit(X_train, y_train)
print("Treinamento concluído!")

# --- Passo 4: Fazer previsões e avaliar o modelo ---
y_pred = ppn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia no conjunto de teste: {accuracy * 100:.2f}%")
erros_teste = (y_test != y_pred).sum()
print(f"Número de erros no conjunto de teste: {erros_teste} de {len(y_test)}")

# --- Passo 5: Visualizar os resultados do modelo ---
print("\nExibindo os resultados do treinamento e a fronteira de decisão...")
plt.figure(figsize=(12, 5))

# Gráfico 1: Erros por época
plt.subplot(1, 2, 1)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Épocas')
plt.ylabel('Número de Erros/Atualizações')
plt.title('Convergência do Perceptron')
plt.grid(True)

# Gráfico 2: Regiões de Decisão
plt.subplot(1, 2, 2)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
test_indices = range(len(y_train), len(y_combined))
plot_decision_regions(X=X_combined, y=y_combined, classifier=ppn, test_idx=test_indices)
plt.xlabel('Comprimento da Sépala [cm]')
plt.ylabel('Comprimento da Pétala [cm]')
plt.title('Fronteira de Decisão do Perceptron')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show() # Exibe o segundo conjunto de gráficos.