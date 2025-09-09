"""
Exemplo Prático: Busca de Similaridade com FAISS
Aula: Bancos de Dados Vetoriais - Inteli Turma 13

Este exemplo demonstra como usar FAISS para:
1. Criar embeddings de texto usando Sentence Transformers
2. Indexar os embeddings no FAISS
3. Realizar buscas por similaridade
4. Aplicar em um caso de uso de busca semântica
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time

def criar_dataset_exemplo():
    """
    Cria um dataset de exemplo com frases sobre tecnologia
    """
    frases = [
        "Machine learning é uma subárea da inteligência artificial",
        "Python é uma linguagem de programação muito popular",
        "Bancos de dados vetoriais armazenam embeddings",
        "Deep learning usa redes neurais profundas",
        "JavaScript é usado para desenvolvimento web",
        "Algoritmos de busca por similaridade são eficientes",
        "Processamento de linguagem natural analisa textos",
        "Sistemas de recomendação sugerem produtos",
        "Inteligência artificial transforma a tecnologia",
        "Desenvolvimento de software requer boas práticas",
        "Análise de dados revela insights importantes",
        "Redes neurais aprendem padrões complexos",
        "Programação orientada a objetos organiza código",
        "Bases de dados relacionais usam SQL",
        "Computação em nuvem oferece escalabilidade"
    ]
    return frases

def criar_embeddings(frases, modelo_nome='all-MiniLM-L6-v2'):
    """
    Cria embeddings das frases usando Sentence Transformers
    
    Args:
        frases: Lista de strings
        modelo_nome: Nome do modelo Sentence Transformers
    
    Returns:
        embeddings: Array numpy com os embeddings
        modelo: Modelo carregado
    """
    print(f"Carregando modelo: {modelo_nome}")
    modelo = SentenceTransformer(modelo_nome)
    
    print("Criando embeddings...")
    embeddings = modelo.encode(frases)
    
    print(f"Embeddings criados: {embeddings.shape}")
    return embeddings, modelo

def criar_indice_faiss(embeddings):
    """
    Cria um índice FAISS para os embeddings
    
    Args:
        embeddings: Array numpy com os embeddings
    
    Returns:
        indice: Índice FAISS criado
    """
    dimensao = embeddings.shape[1]
    print(f"Criando índice FAISS com dimensão: {dimensao}")
    
    # Usando IndexFlatL2 para busca exata (mais lento, mas preciso)
    indice = faiss.IndexFlatL2(dimensao)
    
    # Adicionando os embeddings ao índice
    indice.add(embeddings.astype('float32'))
    
    print(f"Índice criado com {indice.ntotal} vetores")
    return indice

def buscar_similaridade(consulta, modelo, indice, frases, k=3):
    """
    Realiza busca por similaridade
    
    Args:
        consulta: String com a consulta
        modelo: Modelo Sentence Transformers
        indice: Índice FAISS
        frases: Lista original de frases
        k: Número de resultados mais similares
    
    Returns:
        resultados: Lista com os resultados
    """
    print(f"\nBuscando por: '{consulta}'")
    
    # Criar embedding da consulta
    embedding_consulta = modelo.encode([consulta])
    
    # Buscar no índice
    inicio = time.time()
    distancias, indices = indice.search(embedding_consulta.astype('float32'), k)
    tempo_busca = time.time() - inicio
    
    print(f"Busca realizada em {tempo_busca:.4f} segundos")
    print(f"Top {k} resultados mais similares:")
    
    resultados = []
    for i, (distancia, idx) in enumerate(zip(distancias[0], indices[0])):
        resultado = {
            'posicao': i + 1,
            'frase': frases[idx],
            'distancia': distancia,
            'similaridade': 1 / (1 + distancia)  # Conversão para similaridade
        }
        resultados.append(resultado)
        print(f"{i+1}. {frases[idx]} (distância: {distancia:.4f})")
    
    return resultados

def exemplo_completo():
    """
    Executa o exemplo completo de busca com FAISS
    """
    print("=== EXEMPLO PRÁTICO: BUSCA DE SIMILARIDADE COM FAISS ===\n")
    
    # 1. Criar dataset
    frases = criar_dataset_exemplo()
    print(f"Dataset criado com {len(frases)} frases\n")
    
    # 2. Criar embeddings
    embeddings, modelo = criar_embeddings(frases)
    
    # 3. Criar índice FAISS
    indice = criar_indice_faiss(embeddings)
    
    # 4. Realizar buscas de exemplo
    consultas_exemplo = [
        "aprendizado de máquina",
        "desenvolvimento web",
        "análise de dados",
        "inteligência artificial"
    ]
    
    for consulta in consultas_exemplo:
        resultados = buscar_similaridade(consulta, modelo, indice, frases)
        print("-" * 50)

def comparar_indices():
    """
    Demonstra diferentes tipos de índices FAISS
    """
    print("\n=== COMPARAÇÃO DE ÍNDICES FAISS ===\n")
    
    # Criar dataset maior para demonstração
    frases = criar_dataset_exemplo() * 100  # Repetir para ter mais dados
    embeddings, _ = criar_embeddings(frases[:50])  # Usar apenas 50 para exemplo
    
    dimensao = embeddings.shape[1]
    
    # Índice 1: IndexFlatL2 (busca exata)
    print("1. IndexFlatL2 (busca exata):")
    indice_flat = faiss.IndexFlatL2(dimensao)
    indice_flat.add(embeddings.astype('float32'))
    
    # Índice 2: IndexIVFFlat (busca aproximada)
    print("2. IndexIVFFlat (busca aproximada):")
    quantizer = faiss.IndexFlatL2(dimensao)
    indice_ivf = faiss.IndexIVFFlat(quantizer, dimensao, 8)  # 8 clusters
    indice_ivf.train(embeddings.astype('float32'))
    indice_ivf.add(embeddings.astype('float32'))
    indice_ivf.nprobe = 3  # Buscar em 3 clusters
    
    # Comparar tempos de busca
    consulta_teste = embeddings[:1].astype('float32')
    
    # Teste IndexFlatL2
    inicio = time.time()
    _, _ = indice_flat.search(consulta_teste, 5)
    tempo_flat = time.time() - inicio
    
    # Teste IndexIVFFlat
    inicio = time.time()
    _, _ = indice_ivf.search(consulta_teste, 5)
    tempo_ivf = time.time() - inicio
    
    print(f"Tempo IndexFlatL2: {tempo_flat:.6f}s")
    print(f"Tempo IndexIVFFlat: {tempo_ivf:.6f}s")
    print(f"Speedup: {tempo_flat/tempo_ivf:.2f}x")

if __name__ == "__main__":
    # Executar exemplo completo
    exemplo_completo()
    
    # Comparar diferentes índices
    comparar_indices()
    
    print("\n=== EXERCÍCIO PARA OS ALUNOS ===")
    print("1. Modifique o dataset com frases da sua área de interesse")
    print("2. Teste diferentes modelos de embedding")
    print("3. Compare a performance de diferentes índices FAISS")
    print("4. Implemente uma função de avaliação da qualidade dos resultados")

