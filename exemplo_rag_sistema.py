"""
Exemplo Prático: Sistema RAG (Retrieval-Augmented Generation)
Aula: Bancos de Dados Vetoriais - Inteli Turma 13

Este exemplo demonstra como implementar um sistema RAG completo:
1. Indexação de documentos em banco vetorial
2. Recuperação de contexto relevante
3. Geração de respostas usando o contexto recuperado
4. Avaliação da qualidade das respostas
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import time
from typing import List, Dict, Tuple
import re

class SistemaRAG:
    """
    Sistema RAG completo usando bancos vetoriais
    """
    
    def __init__(self, modelo_embedding='all-MiniLM-L6-v2'):
        """
        Inicializa o sistema RAG
        
        Args:
            modelo_embedding: Nome do modelo para criar embeddings
        """
        print(f"Inicializando Sistema RAG com modelo: {modelo_embedding}")
        self.modelo = SentenceTransformer(modelo_embedding)
        self.documentos = []
        self.chunks = []
        self.metadados = []
        self.indice = None
        
    def adicionar_documento(self, texto: str, titulo: str = "", metadados: Dict = None):
        """
        Adiciona um documento ao sistema, dividindo em chunks
        
        Args:
            texto: Conteúdo do documento
            titulo: Título do documento
            metadados: Metadados adicionais
        """
        if metadados is None:
            metadados = {}
            
        # Dividir documento em chunks
        chunks_doc = self._dividir_em_chunks(texto)
        
        doc_id = len(self.documentos)
        self.documentos.append({
            'id': doc_id,
            'titulo': titulo,
            'texto': texto,
            'metadados': metadados
        })
        
        # Adicionar chunks com metadados
        for i, chunk in enumerate(chunks_doc):
            chunk_metadados = {
                'doc_id': doc_id,
                'chunk_id': i,
                'titulo': titulo,
                **metadados
            }
            self.chunks.append(chunk)
            self.metadados.append(chunk_metadados)
            
        print(f"Documento '{titulo}' adicionado com {len(chunks_doc)} chunks")
        
    def _dividir_em_chunks(self, texto: str, tamanho_chunk: int = 500) -> List[str]:
        """
        Divide um texto em chunks menores
        
        Args:
            texto: Texto a ser dividido
            tamanho_chunk: Tamanho aproximado de cada chunk
            
        Returns:
            Lista de chunks
        """
        # Dividir por sentenças primeiro
        sentencas = re.split(r'[.!?]+', texto)
        
        chunks = []
        chunk_atual = ""
        
        for sentenca in sentencas:
            sentenca = sentenca.strip()
            if not sentenca:
                continue
                
            # Se adicionar esta sentença não ultrapassar o limite
            if len(chunk_atual + sentenca) < tamanho_chunk:
                chunk_atual += sentenca + ". "
            else:
                # Salvar chunk atual e começar novo
                if chunk_atual:
                    chunks.append(chunk_atual.strip())
                chunk_atual = sentenca + ". "
        
        # Adicionar último chunk
        if chunk_atual:
            chunks.append(chunk_atual.strip())
            
        return chunks
    
    def criar_indice(self):
        """
        Cria o índice FAISS com todos os chunks
        """
        if not self.chunks:
            raise ValueError("Nenhum documento foi adicionado")
            
        print(f"Criando embeddings para {len(self.chunks)} chunks...")
        embeddings = self.modelo.encode(self.chunks, show_progress_bar=True)
        
        # Criar índice FAISS
        dimensao = embeddings.shape[1]
        self.indice = faiss.IndexFlatL2(dimensao)
        self.indice.add(embeddings.astype('float32'))
        
        print(f"Índice criado com {len(self.chunks)} chunks")
        
    def recuperar_contexto(self, pergunta: str, k: int = 5) -> List[Dict]:
        """
        Recupera chunks relevantes para uma pergunta
        
        Args:
            pergunta: Pergunta do usuário
            k: Número de chunks a recuperar
            
        Returns:
            Lista de chunks relevantes com metadados
        """
        if self.indice is None:
            raise ValueError("Índice não foi criado. Execute criar_indice() primeiro")
            
        # Criar embedding da pergunta
        embedding_pergunta = self.modelo.encode([pergunta])
        
        # Buscar chunks similares
        distancias, indices = self.indice.search(embedding_pergunta.astype('float32'), k)
        
        # Organizar resultados
        contexto = []
        for distancia, idx in zip(distancias[0], indices[0]):
            contexto.append({
                'chunk': self.chunks[idx],
                'metadados': self.metadados[idx],
                'distancia': distancia,
                'similaridade': 1 / (1 + distancia)
            })
            
        return contexto
    
    def gerar_resposta(self, pergunta: str, contexto: List[Dict]) -> str:
        """
        Gera uma resposta baseada no contexto recuperado
        (Simulação - em um sistema real usaria um LLM)
        
        Args:
            pergunta: Pergunta do usuário
            contexto: Contexto recuperado
            
        Returns:
            Resposta gerada
        """
        # Combinar chunks do contexto
        texto_contexto = "\n\n".join([item['chunk'] for item in contexto])
        
        # Simulação de geração de resposta
        # Em um sistema real, isso seria feito por um LLM como GPT, Claude, etc.
        resposta = f"""
Baseado no contexto fornecido, posso responder sua pergunta sobre "{pergunta}":

{texto_contexto[:500]}...

Esta resposta foi gerada com base nos documentos mais relevantes encontrados no banco de dados vetorial.
"""
        return resposta
    
    def responder_pergunta(self, pergunta: str, k: int = 3) -> Dict:
        """
        Pipeline completo: recupera contexto e gera resposta
        
        Args:
            pergunta: Pergunta do usuário
            k: Número de chunks para contexto
            
        Returns:
            Dicionário com pergunta, contexto, resposta e metadados
        """
        print(f"\nProcessando pergunta: '{pergunta}'")
        
        # Recuperar contexto
        inicio = time.time()
        contexto = self.recuperar_contexto(pergunta, k)
        tempo_recuperacao = time.time() - inicio
        
        # Gerar resposta
        inicio = time.time()
        resposta = self.gerar_resposta(pergunta, contexto)
        tempo_geracao = time.time() - inicio
        
        resultado = {
            'pergunta': pergunta,
            'contexto': contexto,
            'resposta': resposta,
            'tempo_recuperacao': tempo_recuperacao,
            'tempo_geracao': tempo_geracao,
            'tempo_total': tempo_recuperacao + tempo_geracao
        }
        
        return resultado
    
    def imprimir_resultado(self, resultado: Dict):
        """
        Imprime o resultado de forma formatada
        """
        print("=" * 80)
        print(f"PERGUNTA: {resultado['pergunta']}")
        print("=" * 80)
        
        print("\nCONTEXTO RECUPERADO:")
        for i, item in enumerate(resultado['contexto']):
            print(f"\n{i+1}. Documento: {item['metadados']['titulo']}")
            print(f"   Similaridade: {item['similaridade']:.4f}")
            print(f"   Chunk: {item['chunk'][:200]}...")
        
        print(f"\nRESPOSTA GERADA:")
        print(resultado['resposta'])
        
        print(f"\nTEMPOS:")
        print(f"- Recuperação: {resultado['tempo_recuperacao']:.4f}s")
        print(f"- Geração: {resultado['tempo_geracao']:.4f}s")
        print(f"- Total: {resultado['tempo_total']:.4f}s")
        print("=" * 80)

def criar_base_conhecimento():
    """
    Cria uma base de conhecimento de exemplo sobre tecnologia
    """
    documentos = [
        {
            'titulo': 'Introdução ao Machine Learning',
            'texto': """
            Machine Learning é uma subárea da inteligência artificial que permite que computadores aprendam e melhorem automaticamente através da experiência, sem serem explicitamente programados. O ML usa algoritmos para analisar dados, identificar padrões e fazer previsões ou decisões.
            
            Existem três tipos principais de machine learning: supervisionado, não supervisionado e por reforço. O aprendizado supervisionado usa dados rotulados para treinar modelos. O não supervisionado encontra padrões em dados sem rótulos. O aprendizado por reforço aprende através de tentativa e erro com um sistema de recompensas.
            
            Algoritmos populares incluem regressão linear, árvores de decisão, redes neurais, SVM e k-means. Cada algoritmo tem suas vantagens e é adequado para diferentes tipos de problemas.
            """,
            'metadados': {'categoria': 'IA', 'nivel': 'iniciante'}
        },
        {
            'titulo': 'Bancos de Dados Vetoriais',
            'texto': """
            Bancos de dados vetoriais são sistemas especializados no armazenamento e consulta de embeddings vetoriais. Eles são fundamentais para aplicações de IA que trabalham com dados não estruturados como texto, imagens e áudio.
            
            Esses bancos usam algoritmos de busca por similaridade para encontrar vetores próximos no espaço multidimensional. Métricas como distância euclidiana, similaridade de cosseno e produto escalar são usadas para medir similaridade.
            
            Ferramentas populares incluem FAISS, Milvus, Weaviate, Pinecone e Chroma. Cada uma tem características específicas de performance, escalabilidade e facilidade de uso.
            """,
            'metadados': {'categoria': 'Banco de Dados', 'nivel': 'intermediario'}
        },
        {
            'titulo': 'Processamento de Linguagem Natural',
            'texto': """
            O Processamento de Linguagem Natural (PLN) é uma área da IA que foca na interação entre computadores e linguagem humana. O objetivo é permitir que máquinas compreendam, interpretem e gerem linguagem natural.
            
            Técnicas de PLN incluem tokenização, análise sintática, reconhecimento de entidades nomeadas, análise de sentimentos e tradução automática. Modelos modernos como BERT, GPT e T5 revolucionaram o campo.
            
            Aplicações práticas incluem chatbots, sistemas de busca semântica, resumo automático de textos, análise de sentimentos em redes sociais e tradução automática.
            """,
            'metadados': {'categoria': 'PLN', 'nivel': 'intermediario'}
        },
        {
            'titulo': 'Sistemas RAG',
            'texto': """
            Retrieval-Augmented Generation (RAG) é uma arquitetura que combina recuperação de informação com geração de texto. O sistema primeiro recupera documentos relevantes de uma base de conhecimento e depois usa essas informações para gerar respostas mais precisas e atualizadas.
            
            O RAG resolve limitações dos modelos de linguagem como conhecimento desatualizado e alucinações. Ele permite que LLMs acessem informações específicas do domínio sem necessidade de retreinamento.
            
            Componentes principais incluem um retriever (geralmente usando embeddings e busca vetorial), um gerador (LLM) e uma base de conhecimento indexada. A qualidade depende tanto da recuperação quanto da geração.
            """,
            'metadados': {'categoria': 'IA', 'nivel': 'avancado'}
        }
    ]
    
    return documentos

def exemplo_sistema_rag_completo():
    """
    Demonstra um sistema RAG completo funcionando
    """
    print("=== SISTEMA RAG COMPLETO ===\n")
    
    # Inicializar sistema
    rag = SistemaRAG()
    
    # Carregar base de conhecimento
    documentos = criar_base_conhecimento()
    
    for doc in documentos:
        rag.adicionar_documento(
            doc['texto'], 
            doc['titulo'], 
            doc['metadados']
        )
    
    # Criar índice
    rag.criar_indice()
    
    # Perguntas de teste
    perguntas = [
        "O que é machine learning?",
        "Como funcionam os bancos de dados vetoriais?",
        "Quais são as aplicações de PLN?",
        "Explique o que é RAG e suas vantagens"
    ]
    
    # Processar cada pergunta
    for pergunta in perguntas:
        resultado = rag.responder_pergunta(pergunta)
        rag.imprimir_resultado(resultado)

def exemplo_avaliacao_rag():
    """
    Demonstra como avaliar a qualidade de um sistema RAG
    """
    print("\n=== AVALIAÇÃO DO SISTEMA RAG ===\n")
    
    rag = SistemaRAG()
    documentos = criar_base_conhecimento()
    
    for doc in documentos:
        rag.adicionar_documento(doc['texto'], doc['titulo'], doc['metadados'])
    
    rag.criar_indice()
    
    # Perguntas com respostas esperadas (para avaliação)
    casos_teste = [
        {
            'pergunta': "Quais são os tipos de machine learning?",
            'resposta_esperada': "supervisionado, não supervisionado, por reforço",
            'categoria_esperada': "IA"
        },
        {
            'pergunta': "Que ferramentas são usadas para bancos vetoriais?",
            'resposta_esperada': "FAISS, Milvus, Weaviate",
            'categoria_esperada': "Banco de Dados"
        }
    ]
    
    print("Avaliando qualidade das respostas:")
    for caso in casos_teste:
        resultado = rag.responder_pergunta(caso['pergunta'], k=2)
        
        # Verificar se a categoria correta foi recuperada
        categorias_recuperadas = [
            item['metadados']['categoria'] 
            for item in resultado['contexto']
        ]
        
        categoria_correta = caso['categoria_esperada'] in categorias_recuperadas
        
        print(f"\nPergunta: {caso['pergunta']}")
        print(f"Categoria esperada: {caso['categoria_esperada']}")
        print(f"Categorias recuperadas: {categorias_recuperadas}")
        print(f"Categoria correta recuperada: {'✓' if categoria_correta else '✗'}")
        print(f"Tempo de resposta: {resultado['tempo_total']:.4f}s")

if __name__ == "__main__":
    # Executar exemplos
    exemplo_sistema_rag_completo()
    exemplo_avaliacao_rag()
    
    print("\n=== EXERCÍCIOS AVANÇADOS ===")
    print("1. Implemente diferentes estratégias de chunking")
    print("2. Adicione re-ranking dos resultados recuperados")
    print("3. Implemente cache para consultas frequentes")
    print("4. Adicione suporte a filtros por metadados")
    print("5. Crie métricas de avaliação automática (BLEU, ROUGE, etc.)")
    print("6. Implemente feedback do usuário para melhorar o sistema")

