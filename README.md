# GNN

Resumo de embeddings: https://github.com/ML-Passionate/GNN/blob/main/EMBEDDINGS.md

Curso LOG disponível no https://www.youtube.com/live/Lj0-Qwzo_B0

Esse código implementa um fluxo completo de processamento de notícias usando **Redes Neurais em Grafos (GNNs)** com o objetivo de identificar e classificar *fake news* usando embeddings de texto e relações de similaridade semântica. Aqui está um **resumo estruturado**:

---

### 🔹 1. **Pré-processamento e Embeddings**

* **Carrega um dataset** contendo textos de notícias e suas classes (`fake` ou `real`).
* Utiliza o modelo `distiluse-base-multilingual-cased` da **Sentence Transformers** para transformar os textos em vetores (`embeddings`).

---

### 🔹 2. **Construção do Grafo**

* Cria um grafo conectando cada notícia aos **3 vizinhos mais próximos** com base na similaridade dos embeddings (usando `kneighbors_graph`).
* Cada nó do grafo contém:

  * `features`: o vetor do embedding.
  * `label`: 0 (fake) ou 1 (real).
  * `text`: conteúdo textual da notícia.

---

### 🔹 3. **Visualização Interativa**

* Usa `Plotly` para exibir a estrutura do grafo.
* Gera layouts com `spring_layout` para posicionar os nós.

---

### 🔹 4. **Graph Autoencoder (GAE)**

* Define um **Autoencoder com convolução em grafos** (`GCNEncoder`) para gerar novos embeddings (representações latentes).
* Treina o autoencoder para reconstruir a estrutura do grafo (links entre nós).
* Visualiza a evolução dos embeddings ao longo das épocas com um GIF animado.

---

### 🔹 5. **One-Class SVM (Aprendizado Não Supervisionado)**

* Utiliza as representações aprendidas pelo GAE para:

  * Treinar um classificador `One-Class SVM` para detectar anomalias (notícias falsas).
  * Avalia o modelo com `classification_report`.
  * Visualiza o resultado com contornos de decisão.

---

### 🔹 6. **Classificação Supervisionada (2 classes) com GCN**

* Treina uma **rede GCN com saída para classificação binária** (Fake vs Real).
* Separa um conjunto de treino e validação.
* Avalia a performance com F1-score macro.
* Exibe gráficos de perda e o grafo final com os embeddings atualizados.

---

### ✅ Objetivo Final:

Combinar NLP e GNN para:

* Gerar embeddings representativos de notícias.
* Construir relações estruturais entre elas.
* Identificar notícias falsas via aprendizado supervisionado e não supervisionado.

---

Deseja que eu gere um fluxograma para ilustrar esse pipeline?



