# Documento de Apresentação de Ferramentas disponíveis, pela Comissão IA UFU

## Considerações Iniciais

Comparação entre LLMs (Large Language Models), Agentes de IA e RAG (Retrieval-Augmented Generation), destacando suas diferenças e aplicações:

---

## 1. LLM (Large Language Model)

### 🔹 O que é?
Modelo de linguagem de grande escala (ex: GPT-4, Gemini, LLaMA) treinado em dados massivos de texto.  
Gera respostas com base em padrões aprendidos, sem acesso direto a fontes externas em tempo real.

### 🔹 Características:
✅ **Generalista**: Pode responder sobre diversos temas, mas com conhecimento estático (até sua data de treinamento).  
✅ **Baseado em probabilidade**: Gera texto plausível, mas pode alucinar (inventar informações incorretas).  
❌ **Sem memória ou busca externa**: Não acessa bancos de dados ou documentos atualizados dinamicamente.

### 🔹 Aplicações:
- Chatbots genéricos.  
- Geração de texto criativo (roteiros, artigos).  
- Tradução automática.

---

## 2. Agentes de IA (AI Agents)

### 🔹 O que é?
Sistemas autônomos que usam LLMs + ferramentas externas para tomar decisões e executar tarefas.  
Podem interagir com APIs, bancos de dados e outros sistemas.

### 🔹 Características:
✅ **Autonomia**: Planeja ações, busca informações e executa tarefas (ex: agendar reuniões, fazer compras online).  
✅ **Memória/Estado**: Mantém contexto ao longo de interações (ex: lembrar preferências do usuário).  
✅ **Integração com ferramentas**: Usa APIs, bancos de dados e até outros modelos de IA.

### 🔹 Aplicações:
- Assistentes pessoais avançados (ex: AutoGPT, BabyAGI).  
- Automação de workflows empresariais.  
- Robôs em jogos ou simulações.

---

## 3. RAG (Retrieval-Augmented Generation)

### 🔹 O que é?
Combina LLM + busca em banco de dados externo para gerar respostas mais precisas e atualizadas.  
Antes de responder, consulta documentos relevantes (ex: manuais, artigos, FAQs).

### 🔹 Características:
✅ **Respostas baseadas em fatos**: Reduz alucinações ao usar fontes confiáveis.  
✅ **Atualização dinâmica**: Pode acessar informações recentes (diferente de um LLM puro).  
❌ **Depende da qualidade da base de dados**: Se a busca retorna dados ruins, a resposta será afetada.

### 🔹 Aplicações:
- Chatbots corporativos (ex: suporte técnico com base em manuais).  
- Sistemas de Q&A sobre documentos específicos (ex: relatórios médicos, jurídicos).  
- Pesquisa acadêmica com citações precisas.

---

## Comparação Resumida

| Recurso               | LLM                  | Agente de IA           | RAG                        |
|------------------------|----------------------|-------------------------|----------------------------|
| **Base de conhecimento** | Treinamento estático | Dinâmico (ferramentas)  | Banco de dados externo     |
| **Atualização em tempo real** | ❌ Não               | ✅ Sim                   | ✅ Sim                     |
| **Capacidade de ação** | ❌ Só gera texto      | ✅ Executa tarefas       | ❌ Só gera texto            |
| **Precisão**           | Variável (pode alucinar) | Depende das ferramentas | Alta (com bons dados)      |
| **Uso típico**         | Chatbots genéricos   | Automação complexa      | Respostas baseadas em documentos |

### Exemplo Prático
- **LLM puro**: "Quem ganhou o Oscar 2024?" → Pode errar se os dados de treinamento forem antigos.  
- **RAG**: Busca em um banco de dados atualizado e responde corretamente.  
- **Agente de IA**: Além de responder, agenda um lembrete para assistir ao filme vencedor no streaming.

---

## Conclusão
- **LLMs** são a base, mas limitados a conhecimento estático.  
- **RAG** melhora a precisão ao adicionar busca em fontes externas.  
- **Agentes de IA** vão além, agindo autonomamente no mundo digital.  
- Cada abordagem tem seu lugar, e combinações (ex: **Agente + RAG**) são poderosas para aplicações avançadas.

---

# 2. Seleção de Sugestões, Ferramentas

## 2.1 Algumas LLM

| Nome         | Link                         | Aplicação / Característica                                                                                  | Observações |
|--------------|------------------------------|-------------------------------------------------------------------------------------------------------------|-------------|
| **ChatGPT**  | [chat.openai.com](https://chat.openai.com) | Desenvolvido pela OpenAI. Ótimo para conversação, geração de texto criativo, programação, resumo, tradução, resposta a perguntas. | Baseado na arquitetura GPT (GPT-3.5, GPT-4). Possui versões gratuitas e pagas (Plus, Team, Enterprise). |
| **Gemini**   | [gemini.google.com](https://gemini.google.com) | Desenvolvido pelo Google. Modelo multimodal (texto, imagem, áudio, vídeo, código). | Integrado a produtos Google. Versões Nano, Pro, Ultra. |
| **Claude**   | [claude.ai](https://claude.ai) | Desenvolvido pela Anthropic. Focado em ser útil, inofensivo e honesto. Bom para tarefas complexas, escrita longa e análise. | Grande janela de contexto e ênfase em segurança e ética. |
| **Llama 3**  | [meta.ai](https://meta.ai) (ou via [Hugging Face](https://huggingface.co)) | Desenvolvido pela Meta. Modelo aberto para pesquisa e desenvolvimento. | Versões com diferentes tamanhos (8B, 70B). |
| **Mistral / Mixtral** | [chat.mistral.ai](https://chat.mistral.ai) | Desenvolvido pela Mistral AI. Foco em eficiência e modelos open-source. | Mixtral usa arquitetura MoE (Mixture-of-Experts). |
| **Command R+** | [cohere.com](https://cohere.com) | Otimizado para RAG e uso empresarial. | Forte em soluções corporativas. |
| **Falcon**   | [huggingface.co/models](https://huggingface.co/models) | Desenvolvido pelo TII (EUA). Modelo open-source em vários tamanhos. | Licença Apache 2.0. |
| **Grok**     | [grok.x.ai](https://grok.x.ai) | Desenvolvido pela xAI. Integração com X/Twitter. | Estilo descontraído e acesso a informações em tempo real. |
| **DeepSeek** | [deepseek.com](https://deepseek.com) | Conhecido por modelos fortes em programação e matemática. | Licenças abertas (MIT, etc.). |

---

## 2.2 Algumas RAG

| Nome        | Link                                | Aplicação / Característica                                                         | Observações |
|-------------|-------------------------------------|------------------------------------------------------------------------------------|-------------|
| **LangChain** | [langchain.com](https://www.langchain.com) | Framework para apps com LLMs + dados externos via RAG. | Compatível com SQL, PDFs, APIs. |
| **LlamaIndex** | [llamaindex.ai](https://www.llamaindex.ai) | Interface entre dados e LLMs para criar RAG. | Conecta bases como Notion e CSV. |
| **Haystack** | [haystack.deepset.ai](https://haystack.deepset.ai) | Framework Python robusto para pipelines RAG. | Suporte a Elasticsearch, FAISS. |
| **Weaviate** | [weaviate.io](https://weaviate.io) | Vetor DB com RAG nativo. | Open-source e escalável. |
| **Pinecone** | [pinecone.io](https://www.pinecone.io) | Vetor DB para recuperação eficiente de contexto. | Foco em performance e escalabilidade. |
| **Vespa**    | [vespa.ai](https://vespa.ai) | Search engine para grandes volumes com RAG. | Usado em ambientes corporativos. |
| **Milvus**   | [milvus.io](https://milvus.io) | Vetor DB open-source de alta performance. | Foco em IA generativa. |
| **Chroma**   | [trychroma.com](https://www.trychroma.com) | Vetor DB leve e simples. | Ideal para POCs. |
| **Jenni AI** | [jenni.ai](https://jenni.ai/) | Ferramenta de escrita acadêmica. | Foco em artigos. |
| **Humata**   | [humata.ai](https://app.humata.ai/) | Respostas baseadas em documentos do usuário. | Foco em leitura de documentos. |
| **Perplexity** | [perplexity.ai](https://www.perplexity.ai/) | Busca avançada com respostas atualizadas. | Foco em leitura de artigos. |
| **NotebookLM** | [notebooklm.google.com](https://notebooklm.google.com/) | Organização e consulta de documentos com IA. | Experimental. |
| **ChatPDF**  | [chatpdf.com](https://www.chatpdf.com) | Interação com PDFs via IA. | Respostas imediatas. |
| **AskYourPDF** | [askyourpdf.com](https://askyourpdf.com) | Perguntas em linguagem natural sobre PDFs. | Extração de informações específicas. |
| **Copilot**  | [copilot.microsoft.com](https://copilot.microsoft.com/chats/b9gnRDRmJLM7irPR77a4C) | Integrado a produtos Microsoft. | Foco em produtividade. |

---

## 2.3 Alguns Agentes de IA

| Nome          | Link                                | Aplicação / Característica                                     | Observações |
|---------------|-------------------------------------|----------------------------------------------------------------|-------------|
| **Auto-GPT**  | [github.com/Torantulino/Auto-GPT](https://github.com/Torantulino/Auto-GPT) | Execução autônoma de tarefas com GPT.                          | Um dos primeiros projetos populares. |
| **AgentGPT**  | [agentgpt.reworkd.ai](https://agentgpt.reworkd.ai) | Interface web para criar agentes GPT.                          | Sem necessidade de codificação. |
| **BabyAGI**   | [github.com/yoheinakajima/babyagi](https://github.com/yoheinakajima/babyagi) | Loop de agente com LLM para execução e reformulação de tarefas.| Focado em objetivos iterativos. |
| **LangGraph** | [langchain.com/langgraph](https://www.langchain.com/langgraph) | Criação de agentes como grafos (multi-passo).                  | Bom para fluxos dinâmicos. |
| **CrewAI**    | [docs.crewai.com](https://docs.crewai.com) | Orquestração de múltiplos agentes com colaboração.             | "Times de agentes". |
| **Superagent**| [superagent.sh](https://superagent.sh) | Plataforma para criar, treinar e hospedar agentes.             | Interface visual + API. |
| **OpenAgents**| *(Em breve no ChatGPT)* | Agentes integrados ao ChatGPT com acesso a ferramentas.         | Recurso em expansão. |
| **Aider**     | [github.com/paul-gauthier/aider](https://github.com/paul-gauthier/aider) | Agente de codificação autônoma (Git + GPT).                    | Produtividade de desenvolvedores. |
| **GitHub Agent** | [smithery.ai/github](https://smithery.ai/server/@smithery-ai/github) | Acesso à API do GitHub (arquivos, repositórios).               | Produtividade de desenvolvedores. |
| **Perplexity Search** | [smithery.ai/perplexity-search](https://smithery.ai/server/@arjunkmrm/perplexity-search) | Pesquisa web com resultados detalhados e citações.             | Foco acadêmico. |
| **PostgreSQL MCP Server** | [smithery.ai/postgresql](https://smithery.ai/server/@HenkDz/postgresql-mcp-server) | Gerenciamento de bancos PostgreSQL com IA.                     | Otimização de operações. |
| **MySQL MCP Server** | [smithery.ai/mysqldb](https://smithery.ai/server/@burakdirin/mysqldb-mcp-server) | Interação de IA com bancos MySQL.                              | Foco em bancos de dados. |

---

# 3. PROMPTs

### 🔹 O que são?
**Prompts** são instruções, perguntas ou descrições usadas para guiar modelos de IA (como LLMs, RAG ou Agentes).  
Um bom prompt é essencial para obter respostas precisas, criativas e contextualmente adequadas.  
Eles podem ser simples (uma única pergunta) ou complexos (estruturados com contexto, exemplos e restrições).

### 🔹 Repositório de Prompts (incluindo leaks)
Mantemos um repositório com **system prompts vazados (leaks)** de diversos modelos, que ajudam a entender como as IAs foram configuradas internamente.  
🔗 [Acesse o repositório aqui](https://github.com/asgeirtj/system_prompts_leaks/)

---

**Observação:** O site [smithery.ai](https://smithery.ai) possui atualmente **3.617 agentes de IA** disponíveis para download.
