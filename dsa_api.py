# Projeto 4 - VisionShop AI: App Para Agente de Vendas com Busca Vetorial e RAG Multimodal (Texto e Imagem)
# Módulo da API

# Importa os para operações no sistema de arquivos 
import os

# Importa a classe principal do FastAPI para criar a aplicação web
from fastapi import FastAPI, UploadFile, File

# Importa BaseModel do Pydantic para modelagem e validação de dados (útil para requests e responses)
from pydantic import BaseModel

# Importa tipos auxiliares para tipagem de estruturas como listas e dicionários
from typing import List, Dict, Any

# Importa shutil para operações de cópia de arquivos e streams (salvar upload no disco)
import shutil

# Importa Collection do Milvus para acessar uma coleção existente no banco vetorial
from pymilvus import Collection

# Importa o wrapper do Ollama para usar um LLM local via LangChain
from langchain_ollama import ChatOllama

# Importa StateGraph e END para construir o fluxo de execução do agente com LangGraph
from langgraph.graph import StateGraph, END

# Importa utilitários e constantes do módulo de banco (embedding e nome da coleção, além da conexão)
from dsa_database import dsa_get_image_embedding, COLLECTION_NAME, connections

# Inicializa a aplicação FastAPI com um título descritivo
app = FastAPI(title = "Visual Search Agent API")

# Inicializa o LLM local (Ollama) com o modelo escolhido e temperatura para controlar criatividade
llm = ChatOllama(model = "llama3", temperature = 0.7)

# Define o tipo de estado que será carregado e atualizado ao longo do grafo do agente
# O estado funciona como memória para o Agente de IA (semelhante a um banco de dados temporário) 
class AgentState(Dict):
    
    # Caminho do arquivo de imagem salvo temporariamente no servidor
    image_path: str
    
    # Lista de produtos encontrados na busca vetorial, com metadados relevantes
    found_products: List[Dict]
    
    # Resposta final do agente de vendas, gerada pelo LLM
    final_response: str

# Nó responsável por gerar embedding da imagem (upload) e realizar a busca no Milvus
def search_node(state: AgentState):
    """Gera embedding da imagem e busca no Milvus"""
    print("--- Buscando Produtos Visualmente Similares ---")
    
    # Garante que há conexão ativa com o Milvus na instância local
    connections.connect("default", host = "localhost", port = "19530")
    
    # Abre a coleção já criada anteriormente no Milvus
    collection = Collection(COLLECTION_NAME)
    
    # Carrega a coleção para permitir buscas 
    collection.load()

    # Gera o embedding vetorial da imagem (upload) apontada pelo caminho no estado
    query_emb = dsa_get_image_embedding(state["image_path"])
    
    # Define parâmetros da busca aproximada, controlando a exploração do índice (nprobe)
    #nprobe: número de clusters a serem verificados durante a busca (se aumentado, melhora a precisão mas aumenta a latência, ou seja, o tempo de resposta da busca)
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    
    # Executa a busca vetorial no campo "embedding" retornando os top K itens e campos desejados
    results = collection.search(
        
        # Envia o vetor de consulta (imagem de upload vetorizada) como lista com um único embedding
        data = [query_emb], 
        
        # Define o campo vetorial onde a busca será feita
        anns_field = "embedding", 
        
        # Passa os parâmetros de busca (métrica e nprobe)
        param = search_params, 
        
        # Limita o número de resultados por consulta
        limit = 2, 
        
        # Define quais campos de metadados retornar junto dos resultados
        output_fields = ["name", "price", "description", "link", "image_path"]
    )

    # Cria uma lista para acumular os produtos encontrados após filtragem
    found = []
    
    # Itera sobre o resultado retornado pelo Milvus (pode haver múltiplas consultas, aqui é 1)
    for hits in results:
        
        # Itera sobre cada item retornado na lista de hits
        for hit in hits:
            
            # Aplica um filtro de similaridade para evitar retornos ruins
            if hit.distance > 0.25: # Ajuste conforme necessário
                
                # Adiciona os dados relevantes do produto e o score no array de resultados
                found.append({
                    
                    # Nome do produto retornado do registro
                    "name": hit.entity.get("name"),
                    
                    # Preço do produto retornado do registro
                    "price": round(hit.entity.get("price"), 2),
                    
                    # Descrição do produto retornada do registro
                    "desc": hit.entity.get("description"),

                    # Link de compra do produto retornado do registro
                    "link": hit.entity.get("link"),

                    # Caminho da imagem do produto no catálogo
                    "image": hit.entity.get("image_path"),

                    # Score de similaridade retornado pela busca
                    "score": hit.distance
                })
    
    # Retorna atualização parcial do estado, contendo os produtos encontrados
    return {"found_products": found}

# Formata um valor numérico no padrão monetário brasileiro (ex.: 10499.0 -> "10.499,00")
def dsa_format_brl(value):
    return f"{value:_.2f}".replace(".", ",").replace("_", ".")

# Nó responsável por gerar um texto de venda usando o LLM a partir dos produtos encontrados
def sales_agent_node(state: AgentState):
    """Usa o LLM para atuar como vendedor"""
    
    print("--- Gerando Argumento de Venda ---")
    
    # Recupera produtos encontrados no estado, com fallback para lista vazia
    products = state.get("found_products", [])
    
    # Caso não haja produtos similares, define um prompt de fallback 
    if not products:
        
        # Prompt para caso de ausência de itens similares no catálogo
        prompt = "O cliente enviou uma foto de um produto, mas não temos nada parecido no estoque. Peça desculpas educadamente e pergunte se ele busca outro estilo."
    
    else:
        
        # Constrói uma string com detalhes dos produtos para alimentar o prompt do LLM
        prod_details = "\n".join([f"- {p['name']} (R$ {dsa_format_brl(p['price'])}): {p['desc']}" for p in products])
        
        # Monta um prompt estruturado com instruções para o LLM agir como vendedor
        prompt = f"""
        Você é um assistente de vendas de moda de alto nível.
        O cliente enviou uma foto e encontramos estes produtos similares no nosso catálogo visual:
        {prod_details}

        1. Confirme que temos produtos parecidos com o da foto.
        2. Descreva brevemente a melhor opção.
        3. Use um tom persuasivo para fechar a venda.
        4. Se houver mais de uma opção, ofereça como alternativa.
        Responda em Português do Brasil.
        """
    
    # Executa o LLM com o prompt e obtém a resposta
    response = llm.invoke(prompt)
    
    # Retorna atualização parcial do estado com a resposta final do agente
    return {"final_response": response.content}

# Cria o grafo tipado pelo estado do agente
workflow = StateGraph(AgentState)

# Registra o nó de busca visual no grafo
workflow.add_node("visual_search", search_node)

# Registra o nó de geração do pitch de vendas no grafo
workflow.add_node("sales_pitch", sales_agent_node)

# Define o nó inicial do fluxo como a busca visual
workflow.set_entry_point("visual_search")

# Conecta o nó de busca visual ao nó de pitch de vendas
workflow.add_edge("visual_search", "sales_pitch")

# Conecta o nó de pitch de vendas ao estado final do grafo
workflow.add_edge("sales_pitch", END)

# Compila o grafo em um agente executável
app_agent = workflow.compile()

# Define um endpoint POST que recebe uma imagem e retorna a resposta do agente e produtos similares
@app.post("/dsa_processa_image")
# Define a função assíncrona que processa o upload do arquivo de imagem (que permite múltiplos uploads, e facilita o processamento pois não bloqueia outras operações)
async def dsa_processa_image(file: UploadFile = File(...)):
    
    # Define um caminho temporário para salvar o arquivo recebido via upload
    file_location = f"temp_{file.filename}"
    
    # Abre um arquivo em modo escrita binária para armazenar o conteúdo recebido
    with open(file_location, "wb") as buffer:
        
        # Copia o stream do upload para o arquivo local
        shutil.copyfileobj(file.file, buffer)
    
    # Monta o estado inicial do agente com a imagem e campos vazios de saída
    inputs = {"image_path": file_location, "found_products": [], "final_response": ""}
    
    # Executa o grafo do agente passando o estado inicial
    result = app_agent.invoke(inputs)
    
    # Remove o arquivo temporário para não acumular arquivos no servidor
    os.remove(file_location)
    
    # Retorna um JSON com a resposta textual do agente e os produtos encontrados
    return {
        
        # Texto final gerado pelo LLM para o cliente
        "response": result["final_response"],
        
        # Lista de produtos similares recuperados na busca vetorial
        "products": result["found_products"]
    }



