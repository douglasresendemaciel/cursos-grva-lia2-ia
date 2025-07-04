# app.py

from flask import Flask, render_template, request
import math
import random
import statistics

app = Flask(__name__)

# =================================================================
# FUNÇÕES DE ÁLGEBRA LINEAR E MATEMÁTICA BÁSICA
# =================================================================

def dot_product(v1, v2):
    """Calcula o produto escalar entre dois vetores."""
    return sum(a * b for a, b in zip(v1, v2))

def matvec_mul(matrix, vector):
    """Multiplica uma matriz por um vetor."""
    return [dot_product(row, vector) for row in matrix]

def vec_add(v1, v2):
    """Soma dois vetores elemento a elemento."""
    return [a + b for a, b in zip(v1, v2)]

# =================================================================
# FUNÇÕES DE ATIVAÇÃO E SUAS DERIVADAS
# =================================================================

def relu(vec):
    """Função de ativação ReLU."""
    return [max(0, v) for v in vec]

def relu_derivative(x):
    """Derivada da função ReLU."""
    return 1 if x > 0 else 0

def sigmoid(x):
    """Função de ativação Sigmoid."""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0 if x < 0 else 1


def sigmoid_derivative(x):
    """Derivada da função Sigmoid."""
    s = sigmoid(x)
    return s * (1 - s)

# =================================================================
# FUNÇÃO DE CUSTO (LOSS) E SUA DERIVADA
# =================================================================

def mse_loss(y_pred, y_true):
    """Calcula o Erro Quadrático Médio (Mean Squared Error)."""
    return (y_pred - y_true)**2

def mse_loss_derivative(y_pred, y_true):
    """Calcula a derivada do Erro Quadrático Médio."""
    return 2 * (y_pred - y_true)

# =================================================================
# FILTRO CUSTOMIZADO PARA O TEMPLATE JINJA2
# =================================================================
def format_vector(vec, precision=4):
    """Formata um vetor (1D) ou uma matriz (2D) para uma string mais legível."""
    # Primeiro, trata o caso de não ser uma lista (ex: um único número como o 'loss')
    if not isinstance(vec, list):
        try:
            return f"{vec:.{precision}f}"
        except (TypeError, ValueError):
            return str(vec)

    # Se a lista estiver vazia, retorna uma lista vazia
    if not vec:
        return "[]"

    # Verifica se é uma matriz (lista de listas)
    is_matrix = isinstance(vec[0], list)

    if is_matrix:
        # Se for uma matriz, formata cada linha (sub-lista)
        formatted_rows = []
        for row in vec:
            # Garante que a linha também seja formatada corretamente
            if isinstance(row, list):
                formatted_row = "  [" + ", ".join(f"{v:.{precision}f}" for v in row) + "]"
                formatted_rows.append(formatted_row)
        # Junta as linhas formatadas com quebras de linha para melhor visualização
        return "[\n" + ",\n".join(formatted_rows) + "\n]"
    else:
        # Se for um vetor simples, formata normalmente
        return "[" + ", ".join(f"{v:.{precision}f}" for v in vec) + "]"


# Registra o filtro para ser usado nos templates (esta linha continua a mesma)
app.jinja_env.filters['format_vector'] = format_vector


# =================================================================
# ROTA PRINCIPAL DA APLICAÇÃO
# =================================================================

@app.route("/", methods=["GET", "POST"])
def index():
    # --- Valores Padrão ---
    input_vector = [0.6, 0.4, 0.9]
    hidden_weights = [[0.1, 0.4, -0.5], [-0.3, 0.2, 0.8], [0.7, -0.9, 0.1]]
    hidden_bias = [0.1, -0.2, 0.05]
    output_weights = [[0.6, -0.1, 0.3]]
    output_bias = [0.05]
    y_true = 0.95  # Valor alvo (correto)
    learning_rate = 0.1

    # --- Se o usuário enviou dados via formulário (POST) ---
    if request.method == "POST":
        input_vector = [float(request.form.get(f"x{i+1}")) for i in range(3)]
        hidden_weights = [[float(request.form.get(f"w{i+1}{j+1}")) for j in range(3)] for i in range(3)]
        hidden_bias = [float(request.form.get(f"b{i+1}")) for i in range(3)]
        output_weights = [[float(request.form.get(f"v{i+1}")) for i in range(3)]]
        output_bias = [float(request.form.get("b_out"))]
        y_true = float(request.form.get("y_true"))
        learning_rate = float(request.form.get("learning_rate"))

    # ========== FORWARD PASS ==========
    # Calcula a saída da rede com os parâmetros atuais
    z_hidden = vec_add(matvec_mul(hidden_weights, input_vector), hidden_bias)
    a_hidden = relu(z_hidden)
    z_output = vec_add(matvec_mul(output_weights, a_hidden), output_bias)
    a_output = [sigmoid(z) for z in z_output]
    y_pred = a_output[0]

    # ========== CÁLCULO DO ERRO ==========
    loss = mse_loss(y_pred, y_true)

    # ========== BACKPROPAGATION ==========
    # Calcula os gradientes usando a Regra da Cadeia, de trás para frente

    # --- 1. Gradientes da Camada de Saída ---
    d_loss_d_a_output = mse_loss_derivative(y_pred, y_true)
    d_a_output_d_z_output = sigmoid_derivative(z_output[0])
    d_loss_d_z_output = d_loss_d_a_output * d_a_output_d_z_output

    grad_output_weights = [[d_loss_d_z_output * val for val in a_hidden]]
    grad_output_bias = [d_loss_d_z_output]

    # --- 2. Gradientes da Camada Oculta ---
    d_loss_d_a_hidden = [d_loss_d_z_output * w for w in output_weights[0]]
    d_loss_d_z_hidden = [d_loss_d_a * relu_derivative(z) for d_loss_d_a, z in zip(d_loss_d_a_hidden, z_hidden)]

    grad_hidden_weights = [[d_loss_dz * inp for inp in input_vector] for d_loss_dz in d_loss_d_z_hidden]
    grad_hidden_bias = d_loss_d_z_hidden

    # ========== ATUALIZAÇÃO DOS PESOS (sugestão) ==========
    # Calcula quais seriam os novos parâmetros após um passo de otimização
    new_output_weights = [[w - learning_rate * gw for w, gw in zip(output_weights[0], grad_output_weights[0])]]
    new_output_bias = [output_bias[0] - learning_rate * grad_output_bias[0]]
    new_hidden_weights = [[w - learning_rate * gw for w, gw in zip(row, grow)] for row, grow in zip(hidden_weights, grad_hidden_weights)]
    new_hidden_bias = [b - learning_rate * gb for b, gb in zip(hidden_bias, grad_hidden_bias)]

    # --- Envia todos os dados para o template renderizar ---
    return render_template("index.html",
        # Parâmetros de entrada
        input_vector=input_vector,
        hidden_weights=hidden_weights,
        hidden_bias=hidden_bias,
        output_weights=output_weights,
        output_bias=output_bias,
        # Resultados do Forward Pass
        z_hidden=z_hidden,
        a_hidden=a_hidden,
        z_output=z_output,
        a_output=a_output,
        # Parâmetros e Resultados do Aprendizado
        y_true=y_true,
        learning_rate=learning_rate,
        loss=loss,
        gradients={
            "output_weights": grad_output_weights,
            "output_bias": grad_output_bias,
            "hidden_weights": grad_hidden_weights,
            "hidden_bias": grad_hidden_bias
        },
        new_params={
            "output_weights": new_output_weights,
            "output_bias": new_output_bias,
            "hidden_weights": new_hidden_weights,
            "hidden_bias": new_hidden_bias
        }
    )


# =================================================================
# ROTA PARA A PÁGINA DE ESTATÍSTICA
# =================================================================
@app.route('/stats', methods=['GET', 'POST'])
def stats_page():
    # --- PARTE 1: Inicialização de Pesos (Lógica existente) ---

    # Valores padrão
    num_weights = 200
    mean_param = 0.0
    std_dev_param = 0.5

    if request.method == 'POST':
        num_weights = int(request.form.get('num_weights'))
        mean_param = float(request.form.get('mean_param'))
        std_dev_param = float(request.form.get('std_dev_param'))

    weights = [random.gauss(mean_param, std_dev_param) for _ in range(num_weights)]
    sample_mean = statistics.mean(weights)
    sample_variance = statistics.variance(weights) if len(weights) > 1 else 0

    # --- PARTE 2: Simulação das Ativações (NOVA LÓGICA) ---
    num_simulations = 1000  # Número de entradas que vamos simular
    pre_activations = []  # Lista para guardar os valores de Z
    activations = []  # Lista para guardar os valores de A (após ReLU)

    # Função para o produto escalar (já deve existir no seu app.py, se não, adicione)
    def dot_product(v1, v2):
        return sum(a * b for a, b in zip(v1, v2))

    for _ in range(num_simulations):
        # Gera uma entrada aleatória com o mesmo número de dimensões que os pesos
        # Usamos uma distribuição uniforme para as entradas, para o resultado ser mais interessante
        random_input = [random.uniform(-1, 1) for _ in range(num_weights)]

        # Calcula a pré-ativação Z (soma ponderada)
        z = dot_product(weights, random_input)
        pre_activations.append(z)

        # Aplica a função de ativação ReLU
        a = max(0, z)
        activations.append(a)

    # --- Envia todos os dados (antigos e novos) para o template ---
    return render_template('stats.html',
                           # Dados da Parte 1
                           num_weights=num_weights,
                           mean_param=mean_param,
                           std_dev_param=std_dev_param,
                           weights_data=weights,
                           sample_mean=sample_mean,
                           sample_variance=sample_variance,
                           # Dados da Parte 2 (NOVOS)
                           pre_activations_data=pre_activations,
                           activations_data=activations)

if __name__ == "__main__":
    app.run(port=5001, host='0.0.0.0', debug=True, threaded=True)  # Use a porta 5001 para evitar conflito com o aula-1/app.py