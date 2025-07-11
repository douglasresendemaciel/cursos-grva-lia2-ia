import logging
from llama_cpp import Llama

# --- Configuração do Logging ---
# Configura o logger para exibir mensagens a partir do nível INFO em um formato claro.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Iniciando o script de interação com o modelo Gemma.")

# --- Definição de Variáveis ---
logging.info("Definindo o caminho para o arquivo do modelo GGUF.")
# Altere "caminho/para/seu/" para o diretório correto em sua máquina
model_path = "Gemma-3-Gaia-PT-BR-4b-it.i1-Q4_K_S.gguf"
pergunta = "Quem é mais forte, Goku ou Vegeta?"

# --- Carregamento do Modelo ---
llm = None
try:
    logging.info(f"Tentando carregar o modelo a partir de: {model_path}")
    # O carregamento pode demorar dependendo do tamanho do modelo e do seu hardware.
    llm = Llama(model_path=model_path, verbose=False)  # verbose=False para não poluir com os logs internos do llama.cpp
    logging.info("Modelo carregado com sucesso na memória.")
except Exception as e:
    logging.error(f"Falha crítica ao carregar o modelo: {e}")
    logging.info("Encerrando o script devido a erro no carregamento.")
    exit()  # Encerra o script se o modelo não puder ser carregado.

# --- Geração de Resposta ---
if llm:
    logging.info("Modelo carregado. Preparando para enviar a pergunta.")
    print("\n" + "=" * 50)
    print(f"**Pergunta:** {pergunta}")
    print("=" * 50 + "\n")

    try:
        logging.info("Enviando prompt para o modelo e aguardando a geração da resposta...")
        output = llm(
            pergunta,
            max_tokens=512,  # Número máximo de tokens a serem gerados. Ajuste conforme necessário.
            echo=False  # Não incluir a pergunta na resposta final.
        )
        logging.info("Resposta recebida do modelo.")

        # --- Processamento e Exibição da Resposta ---
        logging.info("Processando a estrutura de dados da resposta.")
        if "choices" in output and len(output["choices"]) > 0 and "text" in output["choices"][0]:
            resposta = output["choices"][0]["text"].strip()
            logging.info("Resposta extraída com sucesso.")
            print(f"**Resposta do Gemma:**\n{resposta}")
        else:
            logging.warning("A estrutura da resposta do modelo não era a esperada. Nenhuma resposta foi extraída.")
            logging.debug(f"Estrutura recebida: {output}")

    except Exception as e:
        logging.error(f"Ocorreu um erro durante a geração da resposta: {e}")

logging.info("Execução do script concluída.")