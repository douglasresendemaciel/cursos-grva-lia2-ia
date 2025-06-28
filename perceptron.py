import numpy as np


class Perceptron:
    """
    Implementação de um classificador Perceptron.

    Parâmetros
    ------------
    learning_rate : float
      Taxa de aprendizado (entre 0.0 e 1.0). Controla o tamanho do passo
      na atualização dos pesos.
    n_epochs : int
      Número de épocas (passagens sobre o conjunto de treinamento).

    Atributos
    -----------
    weights_ : array-1d
      Pesos após o treinamento.
    bias_ : float
      Termo de viés (bias) após o treinamento.
    errors_ : list
      Número de classificações incorretas (erros) em cada época.
    """

    def __init__(self, learning_rate=0.01, n_epochs=50):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights_ = np.array([])  # Inicializado no método fit
        self.bias_ = 0.0
        self.errors_ = []

    def fit(self, X, y):
        """
        Ajusta o modelo aos dados de treinamento.

        Parâmetros
        ----------
        X : array, shape = [n_samples, n_features]
          Vetor de treinamento, onde n_samples é o número de amostras e
          n_features é o número de características.
        y : array, shape = [n_samples]
          Valores alvo (rótulos da classe).

        Retorna
        -------
        self : object
        """
        # Inicializa os pesos com zeros. O número de pesos é igual ao número de características.
        # O viés (bias) é inicializado com zero.
        self.weights_ = np.zeros(X.shape[1])
        self.bias_ = 0.0
        self.errors_ = []

        # Loop de treinamento pelo número de épocas especificado
        for _ in range(self.n_epochs):
            errors = 0
            # Itera sobre cada amostra de treinamento (X[i]) e seu rótulo (y[i])
            for xi, target in zip(X, y):
                # Faz a previsão para a amostra atual
                prediction = self.predict(xi)

                # Calcula o erro e a atualização dos pesos
                # O erro é (valor_real - valor_previsto)
                update = self.learning_rate * (target - prediction)

                # Atualiza os pesos e o viés
                self.weights_ += update * xi
                self.bias_ += update

                # Incrementa o contador de erros se a atualização não for zero
                errors += int(update != 0.0)

            # Armazena o número de erros da época atual
            self.errors_.append(errors)

        return self

    def net_input(self, X):
        """Calcula a entrada líquida (soma ponderada)"""
        return np.dot(X, self.weights_) + self.bias_

    def predict(self, X):
        """
        Prevê o rótulo da classe.

        Retorna 1 se a entrada líquida for >= 0.0, senão retorna 0.
        Esta é a função de ativação degrau (step function).
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)