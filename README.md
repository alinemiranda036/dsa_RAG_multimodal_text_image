# 🛍️ VisionShop AI: Agente de Vendas com RAG Multimodal (Texto e Imagem)

Um projeto de **RAG Multimodal** que combina busca vetorial de **imagens** com geração de **texto** para criar um assistente de vendas: o usuário envia a foto de um produto, o sistema encontra os itens mais parecidos no catálogo e um LLM local escreve o argumento de venda.

## 📋 Descrição do Projeto

- **Embeddings de imagem (CLIP)**: o modelo `clip-ViT-B-32` transforma cada imagem em um vetor de 512 dimensões
- **Banco Vetorial (Milvus)**: armazena o catálogo e recupera os produtos por similaridade de cosseno
- **Agente de IA (LangGraph)**: orquestra dois passos — busca visual e geração do argumento de venda
- **LLM local (Ollama + Llama 3)**: gera a resposta em português, sem API paga
- **API (FastAPI)**: recebe a imagem e devolve a resposta do agente e os produtos encontrados
- **Frontend (Flask + HTML/Tailwind)**: página de upload e exibição do resultado

## 🎯 Caso de Uso

- 🛒 **E-commerce Visual**: recomendação de produtos a partir de uma foto
- 📸 **Busca de Produtos**: o usuário tira uma foto e o sistema encontra similares no catálogo
- 💼 **Agentes de Vendas Inteligentes**: IA que reconhece o produto e sugere alternativas
- 🔬 **Estudo de IA Multimodal**: embeddings de imagem, busca vetorial e RAG visual

## 🏗️ Arquitetura

```
┌──────────────────────────────────────────────────────┐
│          Imagem enviada pelo usuário                 │
│          (upload na interface web)                   │
└────────────┬─────────────────────────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │  Frontend (Flask)           │  dsa_frontend.py + templates/index.html
      │  http://localhost:5000      │  salva a imagem em static/uploads/
      └──────┬──────────────────────┘
             │  POST multipart
      ┌──────▼──────────────────────┐
      │  API (FastAPI)              │  dsa_api.py
      │  POST /dsa_processa_image   │  http://localhost:8000
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────────────────────────┐
      │  Agente LangGraph                               │
      │                                                 │
      │  1. visual_search  (search_node)                │
      │     ├─ CLIP gera o embedding da imagem (512d)   │
      │     └─ Milvus devolve os 2 mais similares       │
      │                                                 │
      │  2. sales_pitch  (sales_agent_node)             │
      │     └─ Llama 3 (Ollama) escreve o argumento     │
      └──────┬──────────────────────────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │  Resposta do agente         │
      │  + produtos sugeridos       │
      └─────────────────────────────┘
```

## 📁 Estrutura do Projeto

```
dsa_RAG_multimodal_text_image/
├── dsa_api.py              # API FastAPI + agente LangGraph
├── dsa_database.py         # Conexão com o Milvus, schema, carga do catálogo e embeddings
├── dsa_frontend.py         # Frontend Flask
├── templates/
│   └── index.html          # Página da aplicação (HTML + Tailwind via CDN)
├── static/
│   └── img/                # Imagens do catálogo (10 produtos)
│       ├── leather_jacket.jpg
│       ├── floral_dress.jpg
│       ├── sneakers.jpg
│       ├── white_shirt.jpg
│       ├── PlayStation_5.png
│       ├── Xbox_Series_X.png
│       ├── MacBook_Air_M2.png
│       ├── Dell_XPS_13.png
│       ├── Sony_WH-1000XM5.png
│       └── JBL_Flip_6.png
├── docker-compose.yml      # Milvus standalone + etcd
├── requirements.txt        # Dependências Python (versões fixadas)
├── .gitignore              # Arquivos e pastas que não vão para o repositório
└── README.md
```

Tudo o que o projeto precisa para rodar já está no repositório: basta clonar. Duas pastas são criadas automaticamente durante a execução e não fazem parte do repositório (estão no `.gitignore`):

| Pasta | Criada por | Conteúdo |
|-------|-----------|----------|
| `static/uploads/` | `dsa_frontend.py` | Imagens enviadas pelo usuário na interface |
| `volumes/` | `docker-compose up` | Dados persistidos do Milvus e do etcd |

## 🚀 Como Executar

### 1️⃣ Pré-requisitos

| Requisito | Detalhe |
|-----------|---------|
| **Python 3.11 ou superior** | Recomendado 3.13 (versão usada no desenvolvimento). Com 3.10 a instalação falha, pois as versões fixadas de `numpy`, `pandas` e `scipy` exigem 3.11+ |
| **Docker Desktop** (ou Docker Engine + Compose) | Precisa estar aberto e rodando antes do passo 4 |
| **Ollama** | Download em https://ollama.com |
| **Espaço em disco** | Cerca de 10 GB livres: modelo Llama 3 (~4,7 GB), PyTorch e demais bibliotecas, modelo CLIP (~600 MB) e imagens Docker |
| **Memória RAM** | 8 GB é o mínimo; 16 GB deixa a geração da resposta bem mais rápida |
| **Internet** | Necessária na instalação e na primeira execução (download dos modelos) |

### 2️⃣ Clonar e instalar as dependências

```bash
git clone https://github.com/alinemiranda036/dsa_RAG_multimodal_text_image.git
cd dsa_RAG_multimodal_text_image
```

**Opção A — venv**

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Opção B — conda**

```bash
conda create --name visionshop python=3.13
conda activate visionshop
pip install -r requirements.txt
```

A instalação demora alguns minutos por causa do PyTorch.

### 3️⃣ Configurar o Ollama

```bash
ollama pull llama3
ollama list        # confirme que "llama3" aparece na lista
```

No Windows e no macOS o Ollama já fica rodando em segundo plano depois de instalado. No Linux, se o serviço não estiver ativo, rode `ollama serve` em um terminal separado.

### 4️⃣ Subir o Milvus (banco vetorial)

```bash
docker-compose up -d       # em versões recentes: docker compose up -d
```

O Milvus leva de 1 a 2 minutos para ficar pronto. Confira antes de seguir:

```bash
docker ps                              # milvus-standalone e milvus-etcd devem estar "Up"
curl http://localhost:9091/healthz     # deve responder OK
```

### 5️⃣ Carregar o catálogo no Milvus

Execute sempre a partir da raiz do projeto, pois os caminhos das imagens (`static/img/...`) são relativos:

```bash
python dsa_database.py
```

Saída esperada:

```
Iniciando o processo de carga no banco vetorial...

Gerando embeddings e populando banco...

Inserido: Jaqueta de Couro Vintage
Inserido: Vestido Floral de Verão
...
Inserido: JBL Flip 6

Banco de dados carregado!
```

Devem aparecer **10 linhas "Inserido"**. Na primeira execução o modelo CLIP é baixado do Hugging Face, então pode demorar um pouco mais.

> Este script apaga e recria a coleção `products_catalog` a cada execução. Só é preciso rodá-lo de novo se o catálogo mudar.

### 6️⃣ Iniciar a API (Terminal 1 — mantenha aberto)

```bash
uvicorn dsa_api:app --port 8000 --reload
```

Aguarde a mensagem `Application startup complete`. A documentação interativa fica em http://localhost:8000/docs.

### 7️⃣ Iniciar o Frontend (Terminal 2)

Abra um novo terminal, entre na pasta do projeto, **ative o ambiente virtual de novo** e rode:

```bash
python dsa_frontend.py
```

### 8️⃣ Testar

1. Acesse http://localhost:5000
2. Clique na área de upload e envie uma imagem (para um primeiro teste, use uma das imagens de `static/img/`)
3. O resultado esperado é o texto do assistente de vendas e até 2 produtos similares, com imagem, preço e botão "Comprar"

A primeira resposta pode levar de alguns segundos a alguns minutos, dependendo do hardware, porque o Llama 3 roda na sua máquina.

### Resumo da ordem

```
Docker Desktop aberto  →  docker-compose up -d  →  aguardar o healthz
Ollama rodando         →  ollama pull llama3
python dsa_database.py                         (uma vez)
Terminal 1: uvicorn dsa_api:app --port 8000 --reload
Terminal 2: python dsa_frontend.py             →  http://localhost:5000
```

### Encerrar

```bash
# Ctrl+C nos terminais da API e do frontend
docker-compose down
```

Os dados do Milvus ficam em `volumes/`. Apague essa pasta se quiser começar do zero.

## 🔧 Componentes Principais

### 1. `dsa_database.py` — Banco Vetorial

Conecta ao Milvus (`localhost:19530`), carrega o modelo CLIP, cria a coleção e insere o catálogo.

```python
encoder = SentenceTransformer('clip-ViT-B-32')
DIMENSION = 512
COLLECTION_NAME = "products_catalog"
```

**Schema da coleção `products_catalog`:**

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `id` | INT64 | Chave primária, gerada automaticamente |
| `name` | VARCHAR(200) | Nome do produto |
| `price` | FLOAT | Preço |
| `description` | VARCHAR(1000) | Descrição |
| `image_path` | VARCHAR(500) | Caminho da imagem (`static/img/...`) |
| `link` | VARCHAR(1000) | Link de compra |
| `embedding` | FLOAT_VECTOR(512) | Embedding da imagem gerado pelo CLIP |

**Índice vetorial:** `IVF_FLAT`, métrica `COSINE`, `nlist = 128`.

Funções:
- `dsa_init_db()` — remove a coleção se existir, recria o schema e o índice
- `dsa_seed_data(collection)` — gera o embedding de cada imagem do catálogo e insere no Milvus (produtos cuja imagem não é encontrada são ignorados)
- `dsa_get_image_embedding(image_path)` — devolve o vetor de uma imagem; também é usada pela API

### 2. `dsa_api.py` — API e Agente

#### Estado do agente

```python
class AgentState(Dict):
    image_path: str              # Caminho temporário da imagem recebida
    found_products: List[Dict]   # Produtos similares encontrados
    final_response: str          # Resposta gerada pelo LLM
```

#### Nó `visual_search` — função `search_node()`

1. Gera o embedding da imagem enviada
2. Busca no Milvus (`COSINE`, `nprobe = 10`, `limit = 2`)
3. Mantém apenas os resultados com similaridade acima de `0.25`
4. Devolve nome, preço, descrição, link, imagem e score de cada produto

#### Nó `sales_pitch` — função `sales_agent_node()`

1. Monta o prompt com os produtos encontrados (preço formatado em reais)
2. Chama o Llama 3 via Ollama (`temperature = 0.7`)
3. Se nenhum produto passou no filtro de similaridade, o agente pede desculpas e pergunta se o cliente busca outro estilo

#### Endpoint `POST /dsa_processa_image`

Recebe um arquivo no campo `file` (multipart) e devolve:

```json
{
  "response": "Texto gerado pelo agente de vendas...",
  "products": [
    {
      "name": "Jaqueta de Couro Vintage",
      "price": 180.21,
      "desc": "Jaqueta clássica preta estilo motociclista.",
      "link": "https://...",
      "image": "static/img/leather_jacket.jpg",
      "score": 0.93
    }
  ]
}
```

Teste direto, sem o frontend:

```bash
curl -X POST http://localhost:8000/dsa_processa_image -F "file=@static/img/sneakers.jpg"
```

### 3. `dsa_frontend.py` + `templates/index.html` — Frontend

- Rota `/` com GET (abre a página) e POST (recebe a imagem)
- Salva a imagem em `static/uploads/` para exibi-la na tela
- Repassa a imagem para a API em `http://127.0.0.1:8000/dsa_processa_image`
- Renderiza `templates/index.html` com a resposta do agente, os produtos e a imagem enviada
- As imagens dos produtos são servidas direto de `static/img/`

O layout usa Tailwind CSS carregado por CDN, por isso o navegador precisa de internet para exibir a página com estilo.

### 4. `docker-compose.yml` — Infraestrutura

| Serviço | Container | Imagem | Portas |
|---------|-----------|--------|--------|
| `etcd` | `milvus-etcd` | `quay.io/coreos/etcd:v3.5.5` | — |
| `standalone` | `milvus-standalone` | `milvusdb/milvus:v2.3.0` | `19530` (gRPC), `9091` (health) |

O Milvus roda em modo standalone com armazenamento local (`COMMON_STORAGETYPE: local`), sem MinIO.

## 📚 Dependências Principais

| Biblioteca | Versão | Uso |
|-----------|--------|-----|
| `fastapi` | 0.128.0 | Framework da API |
| `uvicorn` | 0.40.0 | Servidor da API |
| `Flask` | 3.1.2 | Frontend web |
| `pymilvus` | 2.6.6 | Cliente do Milvus |
| `sentence-transformers` | 5.2.0 | Modelo CLIP (embeddings) |
| `torch` | 2.10.0 | Backend do modelo CLIP |
| `pillow` | 12.1.0 | Leitura das imagens |
| `langgraph` | 1.0.7 | Orquestração do agente |
| `langchain-ollama` | 1.0.1 | Integração com o LLM local |
| `python-multipart` | 0.0.21 | Upload de arquivos no FastAPI |

Lista completa em `requirements.txt`.

## ⚙️ Personalização

### Adicionar produtos ao catálogo

1. Coloque a imagem em `static/img/`
2. Adicione um item na lista `products` dentro de `dsa_seed_data()`, em `dsa_database.py`:
   ```python
   {"name": "Nome", "price": 99.90, "desc": "Descrição.", "img": "static/img/arquivo.jpg", "link": "https://..."}
   ```
3. Rode `python dsa_database.py` novamente

### Ajustar o número de produtos retornados

Em `dsa_api.py`, função `search_node()`: altere `limit = 2`.

### Ajustar o limite de similaridade

Em `dsa_api.py`, função `search_node()`:

```python
if hit.distance > 0.25:   # valores maiores deixam a busca mais exigente
```

### Trocar o modelo LLM

```python
# Em dsa_api.py (baixe o modelo antes com: ollama pull mistral)
llm = ChatOllama(model = "mistral", temperature = 0.7)
```

### Trocar o modelo de embedding

Em `dsa_database.py`, altere o modelo em `SentenceTransformer(...)` e ajuste `DIMENSION` para o tamanho do vetor do novo modelo. Depois rode `python dsa_database.py` para recriar a coleção.

## 🛡️ Privacidade

- ✅ **Inferência local**: embeddings (CLIP), busca (Milvus) e geração de texto (Llama 3) rodam na sua máquina; nenhuma imagem é enviada para APIs externas
- ℹ️ **Internet**: usada na instalação, no primeiro download dos modelos e para carregar o Tailwind CSS (CDN) no navegador
- ℹ️ **Uploads**: a cópia temporária usada pela API (`temp_<arquivo>`) é apagada após o processamento; a cópia do frontend permanece em `static/uploads/`

## 🐛 Troubleshooting

### `MilvusException` / "connection refused" ao rodar `dsa_database.py` ou a API

O Milvus não está no ar ou ainda está iniciando.

```bash
docker ps
docker-compose up -d
curl http://localhost:9091/healthz
docker-compose logs -f standalone
```

### Erro ao instalar `numpy`, `pandas` ou `scipy`

Versão do Python abaixo de 3.11. Confira com `python --version` e recrie o ambiente com Python 3.11+.

### `python dsa_database.py` termina sem nenhuma linha "Inserido"

O script foi executado fora da raiz do projeto, ou a pasta `static/img/` não está presente. Entre na pasta do repositório e rode de novo.

### "Erro ao conectar com o Agente Inteligente" na tela

A API não está rodando ou falhou. Verifique o Terminal 1: ele precisa mostrar `Application startup complete` e nenhum erro após o upload.

### Erro relacionado ao Ollama (`model 'llama3' not found` ou conexão recusada na porta 11434)

```bash
ollama list          # o modelo llama3 precisa estar na lista
ollama pull llama3
ollama serve         # apenas se o serviço não estiver rodando
```

### A resposta diz que não há nada parecido no estoque

Nenhum produto passou do limite de similaridade (`0.25`). Teste com uma imagem de `static/img/` para confirmar que o fluxo está funcionando.

### Porta 8000 ou 5000 em uso

Se mudar a porta da API, atualize também `API_URL` em `dsa_frontend.py`. A porta do frontend é definida no final de `dsa_frontend.py`. No macOS, a porta 5000 pode estar ocupada pelo AirPlay Receiver.

### `conflict` de nome de container ao subir o Docker

Já existe um container `milvus-standalone` ou `milvus-etcd` de outro projeto. Remova com `docker rm -f milvus-standalone milvus-etcd` e suba novamente.

## 📈 Melhorias Futuras

- [ ] Busca por texto usando o encoder de texto do CLIP (ex.: "tênis vermelho")
- [ ] Filtros por preço e categoria na busca vetorial
- [ ] Sanitizar o nome do arquivo no upload (`secure_filename`) e limpar `static/uploads/`
- [ ] Configuração por variáveis de ambiente (host do Milvus, modelo, URL da API)
- [ ] Catálogo em arquivo ou banco de dados em vez de lista no código
- [ ] Containerizar API e frontend no `docker-compose.yml`
- [ ] Testes automatizados

## 📝 Licença

Projeto de Pós-Graduação, desenvolvido para fins de estudo e portfólio durante a Pós-Graduação em Engenharia de Dados para IA da Data Science Academy. Não possui licença de uso comercial.

## 📞 Contato

Dúvidas ou sugestões sobre o projeto? Entre em contato comigo: **aline.abm97@gmail.com**

## 🎓 Referências

- [Milvus Documentation](https://milvus.io/docs)
- [CLIP ViT-B-32 (sentence-transformers)](https://huggingface.co/sentence-transformers/clip-ViT-B-32)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Ollama](https://github.com/ollama/ollama)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)
