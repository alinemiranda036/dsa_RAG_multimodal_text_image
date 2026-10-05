# 🛍️ VisionShop AI: Agente de Vendas com RAG Multimodal (Texto e Imagem)

Um projeto de **RAG Multimodal** que combina busca vetorial de **imagens** e **texto** para criar um assistente inteligente de vendas que reconhece produtos visualmente e gera argumentos de venda contextualizados.

## 📋 Descrição do Projeto

Este projeto implementa um sistema completo de agente de vendas baseado em visão e IA que:

- **Busca Multimodal**: Combina embeddings de imagem com busca vetorial em banco de dados
- **Banco Vetorial (Milvus)**: Armazena e recupera produtos através de vetores de imagem
- **Reconhecimento Visual**: Processa imagens enviadas pelo usuário para encontrar produtos similares
- **Agente de IA**: Usa LLaMA 3 (via Ollama) para gerar argumentos de venda persuasivos
- **Processamento Local**: Funciona completamente offline sem dependências de APIs externas
- **API FastAPI**: Endpoints prontos para integração em aplicações web

## 🎯 Caso de Uso

O sistema é ideal para:
- 🛒 **E-commerce Visual**: Recomendação de produtos baseada em imagens
- 📸 **Busca de Produtos**: Usuário tira foto e o sistema encontra similares no catálogo
- 💼 **Agentes de Vendas Inteligentes**: IA que reconhece produtos visuais e sugere alternativas
- 🎨 **Catálogos de Moda/Design**: Encontrar produtos com estilos parecidos
- 🔬 **Estudos em IA Multimodal**: Aprender sobre embeddings de imagem e RAG visual

## 🏗️ Arquitetura

```
┌──────────────────────────────────────────────────────┐
│         Imagem Enviada pelo Usuário                 │
│          (upload via API FastAPI)                    │
└────────────┬─────────────────────────────────────────┘
             │
      ┌──────▼──────────┐
      │  Embedding Model│ (Vision - ex: CLIP, ViT)
      │  (Imagem)       │
      └──────┬──────────┘
             │
      ┌──────▼──────────────────────┐
      │  Busca Vetorial (Milvus)    │
      │  - Similaridade de imagens  │
      │  - Top K resultados         │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │  Produtos Recuperados       │
      │  (nome, preço, descrição)   │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │  LangGraph Agent Pipeline   │
      │  ├─ search_node             │
      │  └─ sales_pitch_node        │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │  Ollama (LLaMA 3)           │
      │  Geração de Argumento       │
      │  de Venda                   │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │  Resposta Final ao Cliente  │
      │  + Produtos Sugeridos       │
      └─────────────────────────────┘
```

## 📁 Estrutura do Projeto

```
dsa_RAG_multimodal_text_image/
├── dsa_api.py                     # Aplicação FastAPI com endpoints
├── dsa_database.py                # Configuração do Milvus e embeddings
├── dsa_frontend.py                # Interface Streamlit
├── docker-compose.yml             # Orchestração de containers (Milvus + etcd)
├── requirements.txt               # Dependências Python
├── apsat_static/                 # Imagens do catálogo (não está no repositório)
└── README.md                     # Este arquivo
```

## ⚠️ Importante: Dados Estáticos Não Inclusos

Este repositório **não contém a pasta `apsat_static/`** que armazena as imagens do catálogo de produtos.

### Como adicionar os dados estáticos ao clonar:

```bash
# Clone o repositório
git clone https://github.com/alinemiranda036/dsa_RAG_multimodal_text_image.git
cd dsa_RAG_multimodal_text_image

# IMPORTANTE: você precisa adicionar a pasta de imagens
# A estrutura esperada é:
# apsat_static/
# ├── produtos/
# │   ├── imagem_1.jpg
# │   ├── imagem_2.jpg
# │   └── ...

# Se você tiver acesso aos arquivos, copie-os para:
cp -r /caminho/da/pasta/apsat_static ./

# Ou crie a estrutura manualmente:
mkdir -p apsat_static/produtos
# e coloque as imagens lá
```

Observação: sem essa pasta, o projeto não consegue carregar o catálogo visual usado para comparar produtos por imagem.

## 🚀 Como Executar

### 1️⃣ Pré-requisitos

- Python 3.10+
- Docker e Docker Compose (para Milvus + etcd)
- Ollama instalado e rodando localmente
- Modelo LLaMA 3 baixado no Ollama
- Pasta `apsat_static/produtos/` com imagens de produtos

### 2️⃣ Instalação

```bash
# Clone o repositório
git clone https://github.com/alinemiranda036/dsa_RAG_multimodal_text_image.git
cd dsa_RAG_multimodal_text_image

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

### 3️⃣ Configurar Ollama

```bash
# Instale Ollama de https://ollama.ai
# Execute o serviço
ollama serve

# Em outro terminal, baixe o modelo LLaMA 3
ollama pull llama3
```

### 4️⃣ Iniciar Milvus (Banco Vetorial)

```bash
# Na pasta do projeto
docker-compose up -d

# Verifique se está rodando
curl http://localhost:9091/healthz

# Visualizar logs
docker-compose logs -f
```

### 5️⃣ Preparar as Imagens de Produtos

```bash
# Certifique-se de que você tem a pasta apsat_static com as imagens
ls -la apsat_static/produtos/

# Se não tiver, crie a estrutura e adicione suas imagens
mkdir -p apsat_static/produtos
```

### 6️⃣ Executar a API FastAPI

```bash
# Terminal 1: Executar a API
python dsa_api.py

# A API estará em http://localhost:8000
# Documentação interativa em http://localhost:8000/docs
```

### 7️⃣ (Opcional) Usar Interface Streamlit

```bash
# Terminal 2: Executar o frontend
streamlit run dsa_frontend.py
```

Acesse `http://localhost:8501`

## 🔧 Componentes Principais

### 1. `dsa_database.py` - Configuração do Banco Vetorial

Configura a conexão com Milvus e gerencia embeddings de imagem:

```python
# Conecta ao Milvus
connections.connect("default", host="localhost", port="19530")

# Define o nome da coleção
COLLECTION_NAME = "produtos_multimodal"
```

**Schema da coleção Milvus:**
```
Coleção: produtos_multimodal
├── id: int64 (primary key)
├── embedding: float_vector (dimensão variável)
├── name: varchar (nome do produto)
├── price: float (preço)
├── description: varchar (descrição)
├── link: varchar (URL de compra)
└── image_path: varchar (caminho da imagem)
```

### 2. `dsa_api.py` - Aplicação FastAPI

Define o agente de vendas e endpoints da API:

#### Estado do Agente (`AgentState`)
```python
class AgentState(Dict):
    image_path: str              # Caminho da imagem enviada
    found_products: List[Dict]   # Produtos similares encontrados
    final_response: str          # Resposta gerada pelo LLM
```

#### Nó de Busca Visual: `search_node()`
1. Recebe imagem do usuário
2. Gera embedding da imagem
3. Executa busca vetorial no Milvus
4. Retorna top-2 produtos similares com score de similaridade

```python
results = collection.search(
    data=[query_emb],
    anns_field="embedding",
    param=search_params,
    limit=2,
    output_fields=["name", "price", "description", "link", "image_path"]
)
```

#### Nó de Argumento de Venda: `sales_agent_node()`
1. Recebe produtos encontrados
2. Monta prompt estruturado com detalhes dos produtos
3. Chama LLaMA 3 via Ollama
4. Retorna argumento persuasivo em português

#### Endpoint principal: `POST /dsa_processa_image`
```python
@app.post("/dsa_processa_image")
async def dsa_processa_image(file: UploadFile = File(...)):
    return {
        "response": result["final_response"],
        "products": result["found_products"]
    }
```

### 3. `dsa_frontend.py` - Interface Streamlit

Interface web para testar o agente:
- Upload de imagens
- Visualização dos produtos encontrados
- Exibição da resposta do agente
- Scores de similaridade

### 4. `docker-compose.yml` - Orquestração

Define os serviços necessários:

```yaml
services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.0
    ports:
      - "19530:19530"
      - "9091:9091"
```

## 📚 Dependências Principais

| Biblioteca | Versão | Uso |
|-----------|--------|-----|
| `fastapi` | 0.104+ | Framework web |
| `pymilvus` | 2.3+ | Cliente Milvus |
| `langchain_ollama` | 0.1+ | LLM local |
| `langgraph` | 0.1+ | Orquestração do agente |
| `streamlit` | 1.28+ | Interface web |
| `pillow` | 10+ | Processamento de imagens |
| `sentence-transformers` | 3+ | Embeddings multimodais |

Ver `requirements.txt` para lista completa.

## 🔍 Fluxo Completo de Execução

```
1. Usuário envia imagem de um produto
   ↓
2. FastAPI recebe e salva temporariamente
   ↓
3. Agente LangGraph inicia execução
   ↓
4. search_node():
   - Gera embedding da imagem
   - Busca no Milvus
   - Retorna 2 produtos similares
   ↓
5. sales_agent_node():
   - Monta prompt com produtos
   - Chama LLaMA 3 (Ollama)
   - Gera argumento de venda
   ↓
6. API retorna JSON com:
   - Resposta textual do agente
   - Lista de produtos sugeridos
   - Scores de similaridade
   ↓
7. Arquivo temporário é removido
```

## ⚙️ Configuração Avançada

### Ajustar número de produtos retornados

Em `dsa_api.py`, função `search_node()`:

```python
# Retornar top 5 produtos em vez de 2
limit = 5
```

### Mudar modelo LLM

```python
# Em dsa_api.py
llm = ChatOllama(model="mistral", temperature=0.7)
# ou
llm = ChatOllama(model="neural-chat", temperature=0.5)
```

### Ajustar threshold de similaridade

```python
# Em search_node(), aumentar valor mínimo
if hit.distance > 0.5:  # Era 0.25
    found.append(...)
```

### Mudar dimensão de embedding

Ajuste o modelo de visão usado em `dsa_database.py` para gerar vetores de diferentes tamanhos.

## 🛡️ Segurança e Privacidade

- ✅ **Totalmente Local**: Nenhum dado enviado para APIs externas
- ✅ **Privado**: Imagens e dados ficam apenas no seu servidor
- ✅ **Offline**: Funciona sem conexão à internet
- ✅ **Temporário**: Arquivos de upload são removidos após processamento
- ✅ **Open Source**: Código aberto e auditável

## 🐛 Troubleshooting

### Erro: "Milvus connection refused"
```bash
# Verifique se Milvus está rodando
docker ps | grep milvus

# Se não estiver:
docker-compose up -d

# Verifique health check
curl http://localhost:9091/healthz
```

### Erro: "Model llama3 not found"
```bash
# Puxe o modelo
ollama pull llama3

# Verifique modelos disponíveis
ollama list
```

### Erro: "FileNotFoundError - apsat_static"
```bash
# Certifique-se que a pasta existe
mkdir -p apsat_static/produtos

# Verifique o conteúdo
ls -la apsat_static/produtos/
```

### Erro: "Port 8000 already in use"
```bash
# Use uma porta diferente
python -m uvicorn dsa_api:app --port 8001
```

## 📈 Melhorias Futuras

- [ ] Suportar upload de múltiplas imagens em batch
- [ ] Cache de embeddings de imagens
- [ ] Dashboard de análise de buscas
- [ ] Integração com banco de dados SQL para produtos
- [ ] Suporte a filtros (preço, categoria, marca)
- [ ] Fine-tuning do modelo de visão com seu catálogo
- [ ] Métricas de performance e relevância
- [ ] API REST com autenticação
- [ ] Deployment com Kubernetes
- [ ] Testes automatizados

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:
1. Abrir issues
2. Submeter pull requests
3. Sugerir melhorias

## 📝 Licença

Este projeto é fornecido como exemplo educacional.

## 📞 Suporte

Para dúvidas ou problemas:
- Abra uma issue no GitHub
- Verifique a documentação do Milvus: https://milvus.io
- Verifique a documentação do Ollama: https://ollama.ai
- Verifique LangGraph: https://langchain-ai.github.io/langgraph/

## 🎓 Referências e Recursos

- [Milvus Documentation](https://milvus.io/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Multimodal RAG Patterns](https://docs.llamaindex.ai/en/stable/examples/multi_modal/)
- [Vision Transformers for Embeddings](https://huggingface.co/docs/transformers/tasks/image_feature_extraction)

---

**Desenvolvido com ❤️ para a comunidade de IA aplicada a E-commerce e Vendas**
