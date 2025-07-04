# app.py

# =================================================================
# IMPORTAÇÕES DE BIBLIOTECAS
# =================================================================
# Flask: É o micro-framework web que usamos para construir a aplicação.
# render_template: Função do Flask para renderizar arquivos HTML.
# request: Objeto do Flask que contém os dados de uma requisição HTTP (ex: dados de formulário).
from flask import Flask, render_template, request

# Math: Biblioteca padrão do Python para funções matemáticas, como a exponencial (math.exp).
import math

# Random: Biblioteca para geração de números aleatórios, usada para inicializar pesos e simular dados.
import random

# A biblioteca 'statistics' foi intencionalmente removida para que os cálculos
# de média e variância fossem implementados manualmente (explicitamente).

# =================================================================
# INICIALIZAÇÃO DA APLICAÇÃO FLASK
# =================================================================
# Cria uma instância da aplicação Flask. '__name__' é uma variável especial do Python
# que ajuda o Flask a saber onde procurar por recursos como templates.
app = Flask(__name__)


# =================================================================
# FUNÇÕES DE ÁLGEBRA LINEAR E MATEMÁTICA BÁSICA (EXPANDIDAS)
# =================================================================
# Estas funções são os blocos de construção para as operações da rede neural.
# Elas foram escritas de forma explícita, sem compreensões de lista, para
# facilitar o entendimento do processo passo a passo.
# =================================================================

def dot_product(v1, v2):
    """
    Calcula o produto escalar entre dois vetores (versão explícita).
    O produto escalar é a soma dos produtos dos elementos correspondentes.
    Ex: dot_product([1, 2], [3, 4]) = 1*3 + 2*4 = 11.

    :param v1: O primeiro vetor (lista de números).
    :param v2: O segundo vetor (lista de números).
    :return: Um único número que representa o produto escalar.
    """
    # Inicializa a variável que acumulará o resultado.
    result = 0
    # Itera sobre os índices dos vetores. Supõe-se que v1 e v2 têm o mesmo tamanho.
    for i in range(len(v1)):
        # Multiplica os elementos na mesma posição 'i' de cada vetor e soma ao resultado.
        result += v1[i] * v2[i]
    return result


def matvec_mul(matrix, vector):
    """
    Multiplica uma matriz por um vetor (versão explícita).
    O resultado é um novo vetor, onde cada elemento é o produto escalar
    de uma linha da matriz pelo vetor de entrada.

    :param matrix: A matriz (lista de listas de números).
    :param vector: O vetor (lista de números).
    :return: Um novo vetor que é o resultado da multiplicação.
    """
    # Lista que armazenará o vetor resultante.
    result = []
    # Itera sobre cada 'row' (que é um vetor) na matriz.
    for row in matrix:
        # Calcula o produto escalar da linha atual pelo vetor de entrada.
        sum_ = dot_product(row, vector)
        # Adiciona o resultado escalar à lista de resultado.
        result.append(sum_)
    return result


def vec_add(v1, v2):
    """
    Soma dois vetores elemento a elemento (versão explícita).
    Ex: vec_add([1, 2], [3, 4]) = [1+3, 2+4] = [4, 6].

    :param v1: O primeiro vetor.
    :param v2: O segundo vetor.
    :return: Um novo vetor que é a soma de v1 e v2.
    """
    result = []
    # Itera sobre os índices dos vetores.
    for i in range(len(v1)):
        # Soma os elementos na posição 'i' e adiciona ao resultado.
        result.append(v1[i] + v2[i])
    return result


def vec_sub(v1, v2):
    """
    Subtrai dois vetores elemento a elemento (versão explícita).
    Ex: vec_sub([3, 4], [1, 2]) = [3-1, 4-2] = [2, 2].

    :param v1: O vetor do qual se subtrai (minuendo).
    :param v2: O vetor a ser subtraído (subtraendo).
    :return: Um novo vetor que é a diferença entre v1 e v2.
    """
    result = []
    # Itera sobre os índices dos vetores.
    for i in range(len(v1)):
        # Subtrai os elementos na posição 'i' e adiciona ao resultado.
        result.append(v1[i] - v2[i])
    return result


def scalar_vec_mul(scalar, vec):
    """
    Multiplica um escalar (um número) por um vetor (versão explícita).
    Cada elemento do vetor é multiplicado pelo escalar.
    Ex: scalar_vec_mul(2, [1, 2, 3]) = [2, 4, 6].

    :param scalar: O número a ser multiplicado.
    :param vec: O vetor.
    :return: Um novo vetor com todos os elementos multiplicados pelo escalar.
    """
    result = []
    # Itera sobre cada valor 'v' no vetor.
    for v in vec:
        # Multiplica o valor pelo escalar e adiciona ao resultado.
        result.append(scalar * v)
    return result


def scalar_mat_mul(scalar, matrix):
    """
    Multiplica um escalar por uma matriz (versão explícita).
    Cada elemento da matriz é multiplicado pelo escalar.

    :param scalar: O número a ser multiplicado.
    :param matrix: A matriz.
    :return: Uma nova matriz com todos os elementos multiplicados pelo escalar.
    """
    result = []
    # Itera sobre cada linha da matriz.
    for row in matrix:
        new_row = []
        # Dentro de cada linha, itera sobre cada valor 'v'.
        for v in row:
            # Multiplica o valor pelo escalar.
            new_row.append(scalar * v)
        # Adiciona a nova linha (com valores multiplicados) à matriz resultado.
        result.append(new_row)
    return result


def mat_add(m1, m2):
    """
    Soma duas matrizes elemento a elemento (versão explícita).

    :param m1: A primeira matriz.
    :param m2: A segunda matriz.
    :return: Uma nova matriz que é a soma de m1 e m2.
    """
    result = []
    # Itera sobre as linhas usando um índice 'i'.
    for i in range(len(m1)):
        row = []
        # Itera sobre as colunas (elementos da linha) usando um índice 'j'.
        for j in range(len(m1[0])):
            # Soma os elementos na mesma posição [i][j] de cada matriz.
            row.append(m1[i][j] + m2[i][j])
        result.append(row)
    return result


def mat_sub(m1, m2):
    """
    Subtrai duas matrizes elemento a elemento (versão explícita).

    :param m1: A matriz da qual se subtrai.
    :param m2: A matriz a ser subtraída.
    :return: Uma nova matriz que é a diferença entre m1 e m2.
    """
    result = []
    # Itera sobre as linhas usando um índice 'i'.
    for i in range(len(m1)):
        row = []
        # Itera sobre as colunas usando um índice 'j'.
        for j in range(len(m1[0])):
            # Subtrai os elementos na mesma posição [i][j].
            row.append(m1[i][j] - m2[i][j])
        result.append(row)
    return result


# =================================================================
# FUNÇÕES DE ATIVAÇÃO E SUAS DERIVADAS (EXPANDIDAS)
# =================================================================
# Funções de ativação introduzem não-linearidade na rede, permitindo que ela
# aprenda relações complexas. As derivadas são essenciais para o backpropagation.
# =================================================================

def relu(vec):
    """
    Aplica a função de ativação ReLU (Unidade Linear Retificada) a cada elemento de um vetor.
    A função retorna o próprio valor se ele for positivo, e 0 caso contrário. f(x) = max(0, x).

    :param vec: O vetor de entrada (pré-ativações, Z).
    :return: O vetor de saída (ativações, A).
    """
    result = []
    # Itera sobre cada valor 'v' no vetor de entrada.
    for v in vec:
        # Aplica a condição da ReLU.
        if v > 0:
            result.append(v)
        else:
            result.append(0)
    return result


def relu_derivative(x):
    """
    Calcula a derivada da função ReLU para um único valor.
    A derivada é 1 se x > 0, e 0 caso contrário. É usada no backpropagation.

    :param x: O valor de entrada (pré-ativação, Z).
    :return: 1 ou 0.
    """
    if x > 0:
        return 1
    else:
        return 0


def sigmoid(x):
    """
    Calcula a função de ativação Sigmoid para um único valor.
    A função comprime qualquer valor de entrada para o intervalo (0, 1).
    Fórmula: f(x) = 1 / (1 + e^(-x)). É muito usada em camadas de saída para problemas de classificação binária.

    :param x: O valor de entrada.
    :return: O valor após a aplicação da Sigmoid.
    """
    try:
        # Calcula e^(-x).
        exp_neg_x = math.exp(-x)
        # Retorna o valor da sigmoide.
        return 1 / (1 + exp_neg_x)
    except OverflowError:
        # Se 'x' for muito grande ou pequeno, math.exp pode estourar.
        # Se x é muito pequeno (ex: -1000), e^(-x) é enorme, e o resultado é próximo de 0.
        # Se x é muito grande (ex: 1000), e^(-x) é próximo de 0, e o resultado é próximo de 1.
        if x < 0:
            return 0
        else:
            return 1


def sigmoid_derivative(x):
    """
    Calcula a derivada da função Sigmoid.
    Fórmula: f'(x) = f(x) * (1 - f(x)). É usada no backpropagation.

    :param x: O valor de entrada (pré-ativação, Z).
    :return: O valor da derivada.
    """
    # Calcula o valor da sigmoid para usar na fórmula da derivada.
    s = sigmoid(x)
    return s * (1 - s)


# =================================================================
# FUNÇÃO DE CUSTO (LOSS) E SUA DERIVADA
# =================================================================

def mse_loss(y_pred, y_true):
    """
    Calcula o Erro Quadrático Médio (Mean Squared Error - MSE).
    Mede a "distância" ou "erro" entre a previsão da rede e o valor real.

    :param y_pred: A previsão feita pela rede.
    :param y_true: O valor verdadeiro (alvo).
    :return: O valor do erro.
    """
    # A fórmula é a diferença ao quadrado.
    return (y_pred - y_true) ** 2


def mse_loss_derivative(y_pred, y_true):
    """
    Calcula a derivada do Erro Quadrático Médio.
    Indica a direção e magnitude do erro. É o ponto de partida do backpropagation.

    :param y_pred: A previsão da rede.
    :param y_true: O valor verdadeiro.
    :return: O valor da derivada do erro.
    """
    # A derivada da função (p-t)^2 em relação a 'p' é 2*(p-t).
    return 2 * (y_pred - y_true)


# =================================================================
# FILTRO CUSTOMIZADO PARA O TEMPLATE JINJA2
# =================================================================
def format_vector(vec, precision=4):
    """
    Formata um vetor (1D) ou uma matriz (2D) para uma string mais legível no HTML.
    Isso não afeta os cálculos, apenas a exibição na página web.

    :param vec: Um número, vetor ou matriz.
    :param precision: O número de casas decimais a serem exibidas.
    :return: Uma string formatada.
    """
    # Trata o caso de não ser uma lista (ex: um único número como o 'loss').
    if not isinstance(vec, list):
        try:
            return f"{vec:.{precision}f}"
        except (TypeError, ValueError):
            return str(vec)

    # Se a lista estiver vazia, retorna uma string de lista vazia.
    if not vec:
        return "[]"

    # Verifica se é uma matriz (lista de listas).
    is_matrix = isinstance(vec[0], list)

    if is_matrix:
        # Se for uma matriz, formata cada linha.
        formatted_rows = []
        for row in vec:
            if isinstance(row, list):
                # Expansão para formatar cada elemento da linha.
                formatted_elements = []
                for v in row:
                    formatted_elements.append(f"{v:.{precision}f}")
                # Junta os elementos formatados da linha.
                formatted_row = "  [" + ", ".join(formatted_elements) + "]"
                formatted_rows.append(formatted_row)
        # Junta as linhas formatadas com quebras de linha.
        return "[\n" + ",\n".join(formatted_rows) + "\n]"
    else:
        # Se for um vetor simples, formata seus elementos.
        formatted_elements = []
        for v in vec:
            formatted_elements.append(f"{v:.{precision}f}")
        return "[" + ", ".join(formatted_elements) + "]"


# Registra a função como um filtro no ambiente Jinja2 do Flask.
# Isso permite usar `| format_vector` nos arquivos HTML.
app.jinja_env.filters['format_vector'] = format_vector


# =================================================================
# ROTA PRINCIPAL DA APLICAÇÃO - DEMONSTRAÇÃO DA REDE NEURAL
# =================================================================

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Rota principal que executa e exibe um passo completo (forward e backward)
    de uma pequena rede neural.
    """
    # --- Valores Padrão ---
    # Estes são os valores iniciais da rede quando a página é carregada pela primeira vez.
    input_vector = [0.6, 0.4, 0.9]  # Vetor de entrada para a rede (features).
    hidden_weights = [[0.1, 0.4, -0.5], [-0.3, 0.2, 0.8], [0.7, -0.9, 0.1]]  # Matriz de pesos da camada oculta.
    hidden_bias = [0.1, -0.2, 0.05]  # Vetor de bias da camada oculta.
    output_weights = [[0.6, -0.1, 0.3]]  # Matriz de pesos da camada de saída.
    output_bias = [0.05]  # Bias da camada de saída.
    y_true = 0.95  # O valor real ou 'correto' que a rede deveria prever.
    learning_rate = 0.1  # Taxa de aprendizado: controla o tamanho do passo na atualização dos pesos.

    # --- Se o usuário enviou dados via formulário (POST) ---
    # Este bloco atualiza os valores padrão com os dados enviados pelo usuário.
    if request.method == "POST":
        # Extrai o vetor de entrada do formulário.
        input_vector = []
        for i in range(3):
            input_vector.append(float(request.form.get(f"x{i + 1}")))

        # Extrai os pesos da camada oculta do formulário.
        hidden_weights = []
        for i in range(3):
            row = []
            for j in range(3):
                row.append(float(request.form.get(f"w{i + 1}{j + 1}")))
            hidden_weights.append(row)

        # Extrai os biases da camada oculta.
        hidden_bias = []
        for i in range(3):
            hidden_bias.append(float(request.form.get(f"b{i + 1}")))

        # Extrai os pesos da camada de saída.
        output_weights_row = []
        for i in range(3):
            output_weights_row.append(float(request.form.get(f"v{i + 1}")))
        output_weights = [output_weights_row]

        # Extrai os demais parâmetros.
        output_bias = [float(request.form.get("b_out"))]
        y_true = float(request.form.get("y_true"))
        learning_rate = float(request.form.get("learning_rate"))

    # ========== FORWARD PASS ==========
    # O "Forward Pass" é o processo de fazer uma previsão.
    # Os dados fluem da entrada para a saída através da rede.

    # 1. Cálculo da camada oculta.
    # z = W*x + b (Multiplicação matriz-vetor + soma de bias)
    z_hidden = vec_add(matvec_mul(hidden_weights, input_vector), hidden_bias)
    # a = relu(z) (Aplicação da função de ativação)
    a_hidden = relu(z_hidden)

    # 2. Cálculo da camada de saída.
    # z = V*a + b_out
    z_output = vec_add(matvec_mul(output_weights, a_hidden), output_bias)
    # a_out = sigmoid(z)
    a_output = []
    for z in z_output:
        a_output.append(sigmoid(z))

    # A previsão final (y_pred) é o primeiro (e único) elemento da ativação de saída.
    y_pred = a_output[0]

    # ========== CÁLCULO DO ERRO ==========
    # Mede quão errada foi a previsão da rede.
    loss = mse_loss(y_pred, y_true)

    # ========== BACKPROPAGATION ==========
    # O "Backpropagation" calcula como cada peso e bias contribuiu para o erro total.
    # Ele usa a "Regra da Cadeia" do cálculo para propagar o gradiente do erro
    # de trás para frente na rede.

    # --- 1. Gradientes da Camada de Saída ---
    # dL/da_out: Derivada da função de custo em relação à ativação de saída.
    d_loss_d_a_output = mse_loss_derivative(y_pred, y_true)
    # da_out/dz_out: Derivada da função de ativação (sigmoid) em relação à sua entrada.
    d_a_output_d_z_output = sigmoid_derivative(z_output[0])
    # dL/dz_out: Gradiente do erro em relação à pré-ativação de saída (Regra da Cadeia).
    d_loss_d_z_output = d_loss_d_a_output * d_a_output_d_z_output

    # dL/dV: Gradiente para os pesos da camada de saída. (dL/dz_out * dz_out/dV)
    grad_output_weights_row = []
    for val in a_hidden:
        grad_output_weights_row.append(d_loss_d_z_output * val)
    grad_output_weights = [grad_output_weights_row]
    # dL/db_out: Gradiente para o bias da camada de saída. (dL/dz_out * dz_out/db_out)
    grad_output_bias = [d_loss_d_z_output]

    # --- 2. Gradientes da Camada Oculta ---
    # dL/da_hidden: Propaga o gradiente para a ativação da camada oculta.
    d_loss_d_a_hidden = []
    for w in output_weights[0]:
        d_loss_d_a_hidden.append(d_loss_d_z_output * w)

    # dL/dz_hidden: Propaga o gradiente para a pré-ativação da camada oculta.
    d_loss_d_z_hidden = []
    for i in range(len(d_loss_d_a_hidden)):
        d_loss_d_a = d_loss_d_a_hidden[i]
        z = z_hidden[i]
        d_loss_d_z_hidden.append(d_loss_d_a * relu_derivative(z))

    # dL/dW: Gradiente para os pesos da camada oculta.
    grad_hidden_weights = []
    for d_loss_dz in d_loss_d_z_hidden:
        row = []
        for inp in input_vector:
            row.append(d_loss_dz * inp)
        grad_hidden_weights.append(row)
    # dL/db_hidden: Gradiente para o bias da camada oculta.
    grad_hidden_bias = d_loss_d_z_hidden

    # ========== ATUALIZAÇÃO DOS PESOS (sugestão) ==========
    # Mostra como os parâmetros seriam atualizados usando o gradiente descendente.
    # novo_param = param_antigo - taxa_de_aprendizado * gradiente

    # Novos pesos da camada de saída.
    new_output_weights_row = []
    for i in range(len(output_weights[0])):
        w = output_weights[0][i]
        gw = grad_output_weights[0][i]
        new_output_weights_row.append(w - learning_rate * gw)
    new_output_weights = [new_output_weights_row]

    # Novo bias da camada de saída.
    new_output_bias = [output_bias[0] - learning_rate * grad_output_bias[0]]

    # Novos pesos da camada oculta.
    new_hidden_weights = []
    for i in range(len(hidden_weights)):
        row = []
        for j in range(len(hidden_weights[i])):
            w = hidden_weights[i][j]
            gw = grad_hidden_weights[i][j]
            row.append(w - learning_rate * gw)
        new_hidden_weights.append(row)

    # Novos biases da camada oculta.
    new_hidden_bias = []
    for i in range(len(hidden_bias)):
        b = hidden_bias[i]
        gb = grad_hidden_bias[i]
        new_hidden_bias.append(b - learning_rate * gb)

    # Envia todos os dados (entradas, intermediários, saídas, gradientes) para o template HTML.
    return render_template("index.html",
                           input_vector=input_vector, hidden_weights=hidden_weights,
                           hidden_bias=hidden_bias, output_weights=output_weights,
                           output_bias=output_bias, z_hidden=z_hidden, a_hidden=a_hidden,
                           z_output=z_output, a_output=a_output, y_true=y_true,
                           learning_rate=learning_rate, loss=loss,
                           gradients={"output_weights": grad_output_weights, "output_bias": grad_output_bias,
                                      "hidden_weights": grad_hidden_weights, "hidden_bias": grad_hidden_bias},
                           new_params={"output_weights": new_output_weights, "output_bias": new_output_bias,
                                       "hidden_weights": new_hidden_weights, "hidden_bias": new_hidden_bias}
                           )


# =================================================================
# ROTA PARA A PÁGINA DE ESTATÍSTICA
# =================================================================
@app.route('/stats', methods=['GET', 'POST'])
def stats_page():
    """
    Esta rota demonstra conceitos estatísticos: inicialização de pesos e o efeito
    dessa inicialização na distribuição das ativações dos neurônios.
    """
    # --- Parâmetros Padrão para a Distribuição de Pesos ---
    num_weights = 200  # Quantidade de pesos a serem gerados (simulando um neurônio com muitos inputs).
    mean_param = 0.0  # Média da distribuição normal para inicializar os pesos.
    std_dev_param = 0.5  # Desvio padrão da distribuição normal.

    # Se o usuário enviar novos parâmetros pelo formulário, atualiza os valores.
    if request.method == 'POST':
        num_weights = int(request.form.get('num_weights'))
        mean_param = float(request.form.get('mean_param'))
        std_dev_param = float(request.form.get('std_dev_param'))

    # Gera uma lista de pesos a partir de uma distribuição Gaussiana (Normal).
    weights = []
    for _ in range(num_weights):
        # random.gauss gera um número aleatório com a média e desvio padrão especificados.
        weights.append(random.gauss(mean_param, std_dev_param))

    # --- CÁLCULO EXPLÍCITO DA MÉDIA E VARIÂNCIA (sem a biblioteca 'statistics') ---

    # 1. Cálculo da Média (Mean)
    # A média é a soma de todos os valores dividida pela quantidade de valores.
    sample_mean = 0
    if num_weights > 0:
        total_sum = 0
        # Acumula a soma de todos os pesos.
        for w in weights:
            total_sum += w
        # Divide a soma pela quantidade para obter a média.
        sample_mean = total_sum / num_weights

    # 2. Cálculo da Variância Amostral (Variance)
    # A variância mede a dispersão dos dados em torno da média.
    sample_variance = 0
    # A variância amostral só é definida para n > 1.
    if num_weights > 1:
        sum_of_squared_diffs = 0
        # Itera sobre os pesos para calcular o somatório de (valor - média)^2.
        for w in weights:
            # Calcula a diferença de cada ponto para a média e eleva ao quadrado.
            sum_of_squared_diffs += (w - sample_mean) ** 2
        # Divide pela contagem de amostras menos 1 (graus de liberdade para amostra).
        sample_variance = sum_of_squared_diffs / (num_weights - 1)

    # --- SIMULAÇÃO DAS ATIVAÇÕES ---
    # Esta parte simula como um neurônio com os pesos gerados acima se comportaria
    # ao receber várias entradas aleatórias.
    num_simulations = 1000  # Número de entradas aleatórias que vamos simular.
    pre_activations = []  # Lista para guardar os valores de Z (antes da ativação).
    activations = []  # Lista para guardar os valores de A (após ReLU).

    # O laço principal da simulação.
    for _ in range(num_simulations):
        # Gera uma entrada aleatória com o mesmo número de dimensões dos pesos.
        random_input = []
        for _ in range(num_weights):
            # Usamos uma distribuição uniforme para as entradas.
            random_input.append(random.uniform(-1, 1))

        # Calcula a pré-ativação Z (soma ponderada, ou produto escalar).
        z = dot_product(weights, random_input)
        pre_activations.append(z)

        # Aplica a função de ativação ReLU.
        a = max(0, z)
        activations.append(a)

    # Envia todos os dados gerados e calculados para o template 'stats.html'.
    return render_template('stats.html',
                           num_weights=num_weights, mean_param=mean_param,
                           std_dev_param=std_dev_param, weights_data=weights,
                           sample_mean=sample_mean, sample_variance=sample_variance,
                           pre_activations_data=pre_activations,
                           activations_data=activations)


# =================================================================
# PONTO DE ENTRADA DA APLICAÇÃO
# =================================================================
# O bloco `if __name__ == "__main__":` garante que o servidor de desenvolvimento
# do Flask só seja executado quando o script é chamado diretamente.
if __name__ == "__main__":
    # app.run inicia a aplicação.
    # port=5001: Roda na porta 5001.
    # host='0.0.0.0': Torna o servidor acessível na rede local.
    # debug=True: Ativa o modo de depuração, que reinicia o servidor a cada alteração
    # e fornece mais informações em caso de erro.
    # threaded=True: Permite que o servidor processe múltiplas requisições ao mesmo tempo.
    app.run(port=5001, host='0.0.0.0', debug=True, threaded=True)