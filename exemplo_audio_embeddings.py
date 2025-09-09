"""
Exemplo Prático: Embeddings de Áudio e Speech-to-Text
Aula: Bancos de Dados Vetoriais - Inteli Turma 13

Este exemplo demonstra como:
1. Processar arquivos de áudio
2. Criar embeddings de áudio
3. Simular transcrições de speech-to-text
4. Usar bancos vetoriais para busca semântica em transcrições
5. Implementar um sistema RAG simples para áudio
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import time
from datetime import datetime

class AudioEmbeddingSystem:
    """
    Sistema para gerenciar embeddings de áudio e transcrições
    """
    
    def __init__(self, modelo_nome='all-MiniLM-L6-v2'):
        """
        Inicializa o sistema com um modelo de embeddings
        """
        print(f"Inicializando sistema com modelo: {modelo_nome}")
        self.modelo = SentenceTransformer(modelo_nome)
        self.transcricoes = []
        self.metadados = []
        self.indice = None
        
    def adicionar_transcricao(self, transcricao, metadados=None):
        """
        Adiciona uma transcrição ao sistema
        
        Args:
            transcricao: Texto da transcrição
            metadados: Dicionário com metadados (falante, timestamp, etc.)
        """
        if metadados is None:
            metadados = {}
            
        # Adicionar timestamp se não fornecido
        if 'timestamp' not in metadados:
            metadados['timestamp'] = datetime.now().isoformat()
            
        # Adicionar ID único
        metadados['id'] = len(self.transcricoes)
        
        self.transcricoes.append(transcricao)
        self.metadados.append(metadados)
        
        print(f"Transcrição adicionada: {transcricao[:50]}...")
        
    def criar_indice(self):
        """
        Cria o índice FAISS com as transcrições
        """
        if not self.transcricoes:
            raise ValueError("Nenhuma transcrição foi adicionada")
            
        print("Criando embeddings das transcrições...")
        embeddings = self.modelo.encode(self.transcricoes)
        
        # Criar índice FAISS
        dimensao = embeddings.shape[1]
        self.indice = faiss.IndexFlatL2(dimensao)
        self.indice.add(embeddings.astype('float32'))
        
        print(f"Índice criado com {len(self.transcricoes)} transcrições")
        
    def buscar_transcricoes(self, consulta, k=5):
        """
        Busca transcrições similares à consulta
        
        Args:
            consulta: Texto da consulta
            k: Número de resultados
            
        Returns:
            Lista de resultados com transcrições e metadados
        """
        if self.indice is None:
            raise ValueError("Índice não foi criado. Execute criar_indice() primeiro")
            
        print(f"\nBuscando por: '{consulta}'")
        
        # Criar embedding da consulta
        embedding_consulta = self.modelo.encode([consulta])
        
        # Buscar no índice
        inicio = time.time()
        distancias, indices = self.indice.search(embedding_consulta.astype('float32'), k)
        tempo_busca = time.time() - inicio
        
        print(f"Busca realizada em {tempo_busca:.4f} segundos")
        
        # Organizar resultados
        resultados = []
        for i, (distancia, idx) in enumerate(zip(distancias[0], indices[0])):
            resultado = {
                'posicao': i + 1,
                'transcricao': self.transcricoes[idx],
                'metadados': self.metadados[idx],
                'distancia': distancia,
                'similaridade': 1 / (1 + distancia)
            }
            resultados.append(resultado)
            
        return resultados
        
    def imprimir_resultados(self, resultados):
        """
        Imprime os resultados de busca de forma formatada
        """
        print(f"\nTop {len(resultados)} resultados:")
        print("-" * 80)
        
        for resultado in resultados:
            print(f"{resultado['posicao']}. {resultado['transcricao']}")
            print(f"   Falante: {resultado['metadados'].get('falante', 'Desconhecido')}")
            print(f"   Timestamp: {resultado['metadados'].get('timestamp', 'N/A')}")
            print(f"   Similaridade: {resultado['similaridade']:.4f}")
            print("-" * 80)

def criar_dataset_audio_exemplo():
    """
    Cria um dataset de exemplo simulando transcrições de áudio
    """
    transcricoes_exemplo = [
        {
            'transcricao': "Bom dia, gostaria de saber sobre os produtos de inteligência artificial da empresa",
            'metadados': {'falante': 'Cliente A', 'tipo': 'atendimento', 'duracao': 15.3}
        },
        {
            'transcricao': "O machine learning está revolucionando a forma como analisamos dados",
            'metadados': {'falante': 'Professor Silva', 'tipo': 'aula', 'duracao': 8.7}
        },
        {
            'transcricao': "Precisamos implementar um sistema de recomendação para nossos usuários",
            'metadados': {'falante': 'Gerente TI', 'tipo': 'reuniao', 'duracao': 12.1}
        },
        {
            'transcricao': "Os bancos de dados vetoriais permitem buscas por similaridade muito eficientes",
            'metadados': {'falante': 'Desenvolvedor', 'tipo': 'apresentacao', 'duracao': 18.9}
        },
        {
            'transcricao': "Como posso cancelar minha assinatura do serviço?",
            'metadados': {'falante': 'Cliente B', 'tipo': 'atendimento', 'duracao': 6.2}
        },
        {
            'transcricao': "O processamento de linguagem natural ajuda a entender textos automaticamente",
            'metadados': {'falante': 'Professor Silva', 'tipo': 'aula', 'duracao': 14.5}
        },
        {
            'transcricao': "Vamos discutir a arquitetura do novo sistema de busca semântica",
            'metadados': {'falante': 'Arquiteto Software', 'tipo': 'reuniao', 'duracao': 22.3}
        },
        {
            'transcricao': "Qual é o preço do plano premium?",
            'metadados': {'falante': 'Cliente C', 'tipo': 'atendimento', 'duracao': 4.1}
        },
        {
            'transcricao': "Deep learning usa redes neurais com múltiplas camadas para aprender padrões",
            'metadados': {'falante': 'Professor Silva', 'tipo': 'aula', 'duracao': 16.8}
        },
        {
            'transcricao': "Precisamos melhorar a performance do nosso sistema de busca",
            'metadados': {'falante': 'Gerente TI', 'tipo': 'reuniao', 'duracao': 9.7}
        }
    ]
    
    return transcricoes_exemplo

def exemplo_sistema_rag_audio():
    """
    Demonstra um sistema RAG simples para áudio/transcrições
    """
    print("=== SISTEMA RAG PARA ÁUDIO/TRANSCRIÇÕES ===\n")
    
    # Inicializar sistema
    sistema = AudioEmbeddingSystem()
    
    # Carregar dados de exemplo
    dados = criar_dataset_audio_exemplo()
    
    # Adicionar transcrições ao sistema
    for item in dados:
        sistema.adicionar_transcricao(
            item['transcricao'], 
            item['metadados']
        )
    
    # Criar índice
    sistema.criar_indice()
    
    # Consultas de exemplo
    consultas = [
        "Como funciona inteligência artificial?",
        "Quero cancelar minha conta",
        "Explique sobre machine learning",
        "Problemas com performance do sistema"
    ]
    
    for consulta in consultas:
        resultados = sistema.buscar_transcricoes(consulta, k=3)
        sistema.imprimir_resultados(resultados)
        
        # Simular resposta RAG
        print("RESPOSTA RAG GERADA:")
        contexto = " ".join([r['transcricao'] for r in resultados[:2]])
        print(f"Baseado no contexto encontrado: {contexto[:100]}...")
        print("Uma resposta seria gerada aqui usando um LLM com o contexto recuperado.\n")
        print("=" * 80)

def exemplo_busca_por_falante():
    """
    Demonstra busca filtrada por falante
    """
    print("\n=== BUSCA FILTRADA POR FALANTE ===\n")
    
    sistema = AudioEmbeddingSystem()
    dados = criar_dataset_audio_exemplo()
    
    for item in dados:
        sistema.adicionar_transcricao(item['transcricao'], item['metadados'])
    
    sistema.criar_indice()
    
    # Buscar apenas transcrições de um falante específico
    falante_alvo = "Professor Silva"
    consulta = "aprendizado de máquina"
    
    resultados = sistema.buscar_transcricoes(consulta, k=10)
    
    # Filtrar por falante
    resultados_filtrados = [
        r for r in resultados 
        if r['metadados'].get('falante') == falante_alvo
    ]
    
    print(f"Resultados para '{consulta}' do falante '{falante_alvo}':")
    sistema.imprimir_resultados(resultados_filtrados)

def exemplo_analise_temporal():
    """
    Demonstra análise temporal das transcrições
    """
    print("\n=== ANÁLISE TEMPORAL DE TRANSCRIÇÕES ===\n")
    
    sistema = AudioEmbeddingSystem()
    dados = criar_dataset_audio_exemplo()
    
    # Adicionar timestamps diferentes
    import datetime
    base_time = datetime.datetime.now()
    
    for i, item in enumerate(dados):
        # Simular timestamps diferentes
        timestamp = base_time - datetime.timedelta(days=i)
        item['metadados']['timestamp'] = timestamp.isoformat()
        sistema.adicionar_transcricao(item['transcricao'], item['metadados'])
    
    sistema.criar_indice()
    
    # Buscar e analisar por período
    resultados = sistema.buscar_transcricoes("inteligência artificial", k=5)
    
    print("Análise temporal dos resultados:")
    for resultado in resultados:
        timestamp = resultado['metadados']['timestamp']
        print(f"- {resultado['transcricao'][:50]}... ({timestamp[:10]})")

if __name__ == "__main__":
    # Executar exemplos
    exemplo_sistema_rag_audio()
    exemplo_busca_por_falante()
    exemplo_analise_temporal()
    
    print("\n=== EXERCÍCIOS PROPOSTOS ===")
    print("1. Implemente filtros por tipo de conteúdo (aula, reunião, atendimento)")
    print("2. Adicione suporte a múltiplos idiomas")
    print("3. Crie um sistema de avaliação da qualidade das transcrições")
    print("4. Implemente busca por duração do áudio")
    print("5. Adicione suporte a embeddings de áudio real (não apenas texto)")

