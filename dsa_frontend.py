# Projeto 4 - VisionShop AI: App Para Agente de Vendas com Busca Vetorial e RAG Multimodal (Texto e Imagem)
# Módulo do Frontend

# Importa os para manipular caminhos, criar pastas e lidar com o sistema de arquivos
import os

# Importa requests para fazer chamadas HTTP para a API do agente (FastAPI)
import requests

# Importa componentes do Flask para criar rotas, renderizar HTML e acessar requisições
from flask import Flask, render_template, request, jsonify

# Cria a aplicação Flask
app = Flask(__name__)

# Define a pasta onde as imagens enviadas serão salvas para exibição no frontend
UPLOAD_FOLDER = 'static/uploads'

# Cria a pasta de uploads caso não exista, evitando erro ao salvar arquivos
os.makedirs(UPLOAD_FOLDER, exist_ok = True)

# Define a URL do endpoint FastAPI que processa a imagem e retorna resposta e produtos similares
API_URL = "http://127.0.0.1:8000/dsa_processa_image"

# Define a rota principal da app, aceitando GET (carregar página) e POST (enviar imagem)
@app.route('/', methods = ['GET', 'POST'])
# Define a função que controla a página inicial e o fluxo de upload
def index():
    
    # Inicializa a variável que armazenará a resposta do agente para exibir no frontend
    agent_response = None
    
    # Inicializa a lista de produtos encontrados para exibir no frontend
    products_found = []
    
    # Inicializa a variável com o caminho da imagem enviada para exibir no frontend
    uploaded_image = None

    # Verifica se a requisição é POST (usuário enviou o formulário com imagem)
    if request.method == 'POST':
        
        # Valida se o campo 'file' existe no formulário enviado
        if 'file' not in request.files:
            
            # Retorna uma mensagem simples caso nenhum arquivo tenha sido incluído no request
            return "Nenhum arquivo enviado"
        
        # Recupera o objeto de arquivo enviado pelo usuário
        file = request.files['file']
        
        # Valida se o nome do arquivo está vazio (usuário não selecionou arquivo)
        if file.filename == '':
            
            # Retorna uma mensagem simples caso o usuário não tenha selecionado arquivo
            return "Nenhum arquivo selecionado"

        # Confirma que existe um arquivo válido para processamento
        if file:
            
            # Inicia o bloco que salva a imagem localmente para exibição no frontend
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            
            # Salva o arquivo no disco
            file.save(filepath)
            
            # Guarda o caminho para que o template possa renderizar a imagem enviada
            uploaded_image = filepath

            # Inicia o bloco que envia a imagem para a API do agente no FastAPI
            with open(filepath, 'rb') as f:
                
                # Monta o payload multipart no formato esperado pelo FastAPI
                files = {'file': (file.filename, f, file.content_type)}
                
                # Envolve a chamada HTTP em try/except para capturar falhas de rede e parsing
                try:
                    
                    # Faz a requisição POST para o endpoint do agente enviando o arquivo
                    response = requests.post(API_URL, files = files)
                    
                    # Converte o corpo da resposta em JSON
                    data = response.json()
                    
                    # Extrai a resposta textual do agente do JSON retornado
                    agent_response = data.get("response")
                    
                    # Extrai a lista de produtos encontrados do JSON retornado
                    products_found = data.get("products")
                
                except Exception as e:
                    
                    # Em caso de erro, prepara uma mensagem para exibir ao usuário
                    agent_response = f"Erro ao conectar com o Agente Inteligente: {str(e)}"

    # Renderiza o template HTML com as variáveis calculadas para apresentar resultado ao usuário
    return render_template('index.html', 
                         
                         # Injeta a resposta do agente no template
                         response = agent_response, 
                         
                         # Injeta a lista de produtos no template
                         products = products_found, 
                         
                         # Injeta o caminho da imagem enviada para exibição
                         image = uploaded_image)

# Verifica se o script está sendo executado diretamente e não importado como módulo
if __name__ == '__main__':
    
    # Inicia o servidor Flask em modo debug na porta 5000
    app.run(debug = True, port = 5000)



