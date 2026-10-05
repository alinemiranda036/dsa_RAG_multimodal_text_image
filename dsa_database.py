# Projeto 4 - VisionShop AI: App Para Agente de Vendas com Busca Vetorial e RAG Multimodal (Texto e Imagem)
# Módulo do Banco Vetorial

# Importa o módulo os para interação com o sistema de arquivos
import os

# Importa o módulo warnings para controlar avisos Python
import warnings

# Importa o módulo logging para controle de logs da aplicação
import logging

# Importa o sistema de logging específico da biblioteca transformers
from transformers import logging as hf_logging

# Reduz a verbosidade dos logs do transformers para apenas erros
#O objetivo é evitar mensagens de aviso desnecessárias durante a execução do código
hf_logging.set_verbosity_error()

# Ignora avisos relacionados ao uso de processadores de imagem lentos
#O objetivo é filtrar para retornar apenas mensagens de aviso diferentes de "Using a slow image processor"
warnings.filterwarnings("ignore", message=".*Using a slow image processor.*")

# Importa componentes do Milvus para conexão, schema e operações vetoriais
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Importa o SentenceTransformer para geração de embeddings multimodais
from sentence_transformers import SentenceTransformer

# Importa a biblioteca PIL para manipulação de imagens
from PIL import Image

# Estabelece conexão com o servidor Milvus local
connections.connect("default", host = "localhost", port = "19530")

# Inicializa o modelo CLIP pré treinado para gerar embeddings de imagem e texto
# https://huggingface.co/sentence-transformers/clip-ViT-B-32
encoder = SentenceTransformer('clip-ViT-B-32')

# Define a dimensão fixa dos vetores de embedding gerados pelo modelo
DIMENSION = 512

# Define o nome da coleção que armazenará o catálogo de produtos
COLLECTION_NAME = "products_catalog"

# Função responsável por inicializar e configurar o banco vetorial
def dsa_init_db():

    # Verifica se a coleção já existe no Milvus
    if utility.has_collection(COLLECTION_NAME):
        
        # Remove a coleção existente para recriação limpa
        utility.drop_collection(COLLECTION_NAME)

    # Define os campos e tipos de dados do schema da coleção
    fields = [
        
        # Campo identificador único com geração automática de IDs
        FieldSchema(name = "id", dtype = DataType.INT64, is_primary = True, auto_id = True),
        
        # Campo para nome do produto em formato texto
        FieldSchema(name = "name", dtype = DataType.VARCHAR, max_length = 200),
        
        # Campo para armazenar o preço do produto
        FieldSchema(name = "price", dtype = DataType.FLOAT),
        
        # Campo para descrição textual do produto
        FieldSchema(name = "description", dtype = DataType.VARCHAR, max_length = 1000),
        
        # Campo que guarda o caminho da imagem do produto
        FieldSchema(name = "image_path", dtype = DataType.VARCHAR, max_length = 500),

        # Campo que guarda o link de compra do produto
        FieldSchema(name = "link", dtype = DataType.VARCHAR, max_length = 1000),

        # Campo vetorial para armazenar o embedding da imagem
        FieldSchema(name = "embedding", dtype = DataType.FLOAT_VECTOR, dim = DIMENSION)
    ]

    # Cria o schema da coleção com uma descrição semAG geral
    schema = CollectionSchema(fields, "Catálogo de Produtos Multimodal")

    # Cria a coleção no Milvus usando o schema definido
    collection = Collection(COLLECTION_NAME, schema)
    
    # Define os parâmetros do índice vetorial para busca por similaridade
    # Esse bloco define o equilíbrio entre precisão, latência e custo computacional da busca vetorial.
    # metric_type: como a similaridade é medida (cosseno)
    # index_type: estratégia de busca (IVF = agrupa em clusters; FLAT = vetores sem compressão)
    # nlist: número de clusters do índice (se aumentado, melhora a precisão mas aumenta o custo computacional)
    #nprobe: número de clusters a serem verificados durante a busca (se aumentado, melhora a precisão mas aumenta a latência, ou seja, o tempo de resposta da busca) - dsa_api.py linha 68
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }

    # Cria o índice vetorial no campo de embeddings
    collection.create_index(field_name = "embedding", index_params = index_params)

    # Retorna a coleção pronta para uso
    return collection

# Função responsável por popular o banco com dados iniciais
def dsa_seed_data(collection):
    
    # Define um conjunto fictício de produtos do catálogo
    products = [
        {"name": "Jaqueta de Couro Vintage", "price": 180.21, "desc": "Jaqueta clássica preta estilo motociclista.", "img": "static/img/leather_jacket.jpg", "link": "https://www.amazon.com.br/WSLCN-Jaqueta-masculina-sint%C3%A9tico-jaqueta/dp/B0FP4XBNW7"},
        {"name": "Vestido Floral de Verão", "price": 159.99, "desc": "Vestido leve com estampa de flores amarelas.", "img": "static/img/floral_dress.jpg", "link": "https://www.amazon.com.br/Vestido-Longuete-Casual-Elegante-Evangelico/dp/B0HKLG5VYN"},
        {"name": "Tênis de Corrida High-Tech", "price": 530.65, "desc": "Tênis esportivo com amortecimento avançado.", "img": "static/img/sneakers.jpg", "link": "https://www.amazon.com.br/masculino-Skechers-Energy-Afterburn-cadar%C3%A7o/dp/B000PRJS4M"},
        {"name": "Camisa Social Branca", "price": 78.90, "desc": "Camisa de algodão egípcio para escritório.", "img": "static/img/white_shirt.jpg", "link": "https://www.amazon.com.br/Camisa-Social-Masculina-Manga-Longa/dp/B0DXLC3GZS"},
        {"name": "PlayStation 5", "price": 4299.90, "desc": "Console de última geração com suporte a 4k e Ray Tracing.", "img": "static/img/PlayStation_5.png", "link": "https://www.amazon.com.br/Sony-PlayStation-Edi%C3%A7%C3%A3o-Digital-Controle/dp/B0GWNKJDCZ"},
        {"name": "Xbox Series X", "price": 6599.99, "desc": "O console mais poderoso da Microsoft, rápido e compatível com GamePass.", "img": "static/img/Xbox_Series_X.png", "link": "https://www.amazon.com.br/Xbox-MSEP200692-X-1TB-Digital/dp/B0DCD9LGHY"},
        {"name": "MacBook Air M2", "price": 10499.00, "desc": "Notebook leve, bateria de longa duração e chip M2 ultra rápido.", "img": "static/img/MacBook_Air_M2.png", "link": "https://www.amazon.com.br/2022-Apple-MacBook-laptop-chip/dp/B0DLJG5Z63"},
        {"name": "Dell XPS 13", "price": 14755.04, "desc": "Notebook Windows premium com tela infinita e alta portabilidade.", "img": "static/img/Dell_XPS_13.png", "link": "https://www.amazon.com.br/Dell-9350-Bluetooth-Thunderbolt-retroiluminado/dp/B0HF6FMJTX"},
        {"name": "Sony WH-1000XM5", "price": 1680.00, "desc": "Fones de ouvido com o melhor cancelamento de ruído do mercado.", "img": "static/img/Sony_WH-1000XM5.png", "link": "https://www.amazon.com.br/Sony-WH-1000XM5-cancelamento-otimizador-cristalinas/dp/B09XS7JWHH"},
        {"name": "JBL Flip 6", "price": 649.00, "desc": "Caixa de som bluetooth portátil e resistente à água.", "img": "static/img/JBL_Flip_6.png", "link": "https://www.amazon.com.br/Caixa-Bluetooth-JBL-Flip-Vermelha/dp/B09HGPKY7Q"}
    ]
    
    # Estrutura auxiliar reservada para inserção de dados em lote
    data_to_insert = [[], [], [], [], []] 
    
    print("\nGerando embeddings e populando banco...\n")
    
    # Itera sobre cada produto do catálogo
    for p in products:
        
        # Verifica se a imagem do produto existe no caminho informado
        if os.path.exists(p['img']):
            
            # Gera o embedding da imagem do produto
            emb = dsa_get_image_embedding(p['img'])
            
            # Insere os dados do produto e o embedding na coleção
            collection.insert([[p['name']], [p['price']], [p['desc']], [p['img']], [p['link']], [emb]])
            
            # Exibe confirmação da inserção do produto
            print(f"Inserido: {p['name']}")
    
    # Garante que os dados inseridos sejam persistidos no Milvus
    collection.flush()
    
    print("\nBanco de dados carregado!\n")

# Função responsável por gerar o embedding vetorial de uma imagem
def dsa_get_image_embedding(image_path):
    
    # Abre a imagem a partir do caminho informado
    img = Image.open(image_path)
    
    # Gera o embedding da imagem usando o modelo CLIP e converte para lista
    return encoder.encode(img).tolist()
    
# Verifica se o script está sendo executado diretamente
if __name__ == "__main__":

    print("\nIniciando o processo de carga no banco vetorial...")
    
    # Inicializa o banco vetorial e a coleção
    coll = dsa_init_db()
    
    # Popula a coleção com os dados iniciais
    dsa_seed_data(coll)




