# EMBENDDIGS

## 🔷 O que é um *embedding*?

![image](https://github.com/user-attachments/assets/abaed47c-4c99-4d31-aa86-cd89ce1e32b4)

Um **embedding** é uma representação vetorial densa e de baixa dimensão de um dado (texto, imagem, tempo, etc.), projetada para **capturar semelhanças semânticas ou estruturais** em um espaço numérico contínuo.

* Transforma dados complexos → em vetores numéricos.
* Vetores semelhantes representam **itens semelhantes**.
* Muito usado em **NLP, machine learning, busca semântica, recomendação e clustering**.

---

## 🔷 Tipos principais de embeddings (por tipo de dado)

| Tipo de Dado         | Exemplos                        | Técnicas / Modelos comuns                 |
| -------------------- | ------------------------------- | ----------------------------------------- |
| **Texto**            | Palavras, frases, documentos    | Word2Vec, BERT, Sentence-BERT, GPT        |
| **Imagem**           | Fotos, gráficos                 | ResNet, CLIP, DINO, EfficientNet          |
| **Áudio**            | Voz, música, sons               | Wav2Vec, Whisper, YAMNet                  |
| **Vídeo**            | Vídeos curtos ou longos         | TimeSformer, ViViT                        |
| **Código**           | Código-fonte                    | CodeBERT, CodeT5                          |
| **Tabelas**          | Dados estruturados (tipo CSV)   | Entity embeddings, TabNet, TabTransformer |
| **Multimodal**       | Combinação (ex: texto + imagem) | CLIP, GPT-4V, Gemini, FLAVA               |
| **Séries Temporais** | Dados sequenciais no tempo      | TS2Vec, LSTM autoencoders, T2Vec          |

---

## 🔷 Como os embeddings são usados?

1. **Busca semântica** — Ex: encontrar documentos ou ações parecidas.
2. **Clustering / agrupamento** — Ex: identificar grupos de ativos com comportamento similar.
3. **Classificação** — Embeddings como entrada para modelos supervisionados.
4. **Redução de dimensionalidade / visualização** — Usando t-SNE, UMAP.
5. **Recomendação** — Ex: sugerir ações, produtos ou estratégias com base em perfil vetorial.

---

## 🔷 Embeddings para dados financeiros, trades e séries temporais

### 1. **Séries Temporais Financeiras**

🔹 Ex: preços de ações, volume, indicadores técnicos
🔹 Técnicas:

* `TS2Vec`, `T2Vec` – geram embeddings de janelas temporais
* **Autoencoders LSTM**
* **Transformers temporais**

📌 Usos: clusterizar ativos, prever regimes de mercado, busca por padrões.

---

### 2. **Dados de Trade e Book de Ordens**

🔹 Ex: logs de compra/venda, ordens, execução
🔹 Técnicas:

* **Entity Embeddings** para colunas categóricas
* **TabNet**, **SAINT**, **TabTransformer** para dados estruturados

📌 Usos: prever lucratividade de trades, modelar comportamento de mercado.

---

### 3. **Textos Financeiros**

🔹 Ex: relatórios de empresas, notícias, balanços
🔹 Modelos:

* **FinBERT**, **BloombergGPT**, **FinancialBERT**

📌 Usos: análise de sentimento, risco, embasamento para decisões quantitativas.

---

### 4. **Multimodal Financeiro**

🔹 Combina preço + texto + indicadores
🔹 Técnicas: redes que integram embeddings de diferentes tipos

📌 Usos: modelos mais ricos para predição ou análise de ativos.

---

## ✅ **Principais modelos de embeddings** (texto, imagens, séries temporais)

### 🔹 **Textuais – Hugging Face / SentenceTransformers**

| Nome do Modelo                          | Tipo                  | Idioma         | Comentário                                     |
| --------------------------------------- | --------------------- | -------------- | ---------------------------------------------- |
| `all-MiniLM-L6-v2`                      | Frases                | 🇺🇸 Inglês    | Leve e rápido, muito usado                     |
| `all-mpnet-base-v2`                     | Frases                | 🇺🇸 Inglês    | Alta performance em similaridade semântica     |
| `paraphrase-multilingual-MiniLM-L12-v2` | Frases                | 🌍 Multilíngue | Compacto e versátil                            |
| `paraphrase-multilingual-mpnet-base-v2` | Frases                | 🌍 Multilíngue | Alta qualidade                                 |
| `bert-base-uncased`                     | Palavras              | 🇺🇸 Inglês    | Base do BERT                                   |
| `sentence-transformers/gtr-t5-base`     | Texto para busca      | 🇺🇸 Inglês    | Muito usado em ranking e busca semântica       |
| `hkunlp/instructor-xl`                  | Embeddings com tarefa | 🇺🇸 Inglês    | Usa instruções no input para embeddings        |
| `text-embedding-ada-002` (OpenAI)       | Texto geral           | 🌍 Multi       | API da OpenAI, alta qualidade e multilinguagem |

---

### 🖼️ **Imagem – via Hugging Face**

| Nome do Modelo                 | Tipo           | Comentário                                |
| ------------------------------ | -------------- | ----------------------------------------- |
| `openai/clip-vit-base-patch32` | Imagem + texto | Multimodal (imagem/texto no mesmo espaço) |
| `facebook/dino-vits8`          | Imagem         | Auto-supervisionado, ótima generalização  |

---

### 🧠 **Séries Temporais (timeseries embeddings)**

| Nome do Modelo           | Tipo                     | Comentário                                     |
| ------------------------ | ------------------------ | ---------------------------------------------- |
| `TS2Vec`                 | Série temporal           | SOTA para timeseries embeddings                |
| `T2Vec`                  | Série temporal           | Similar ao Word2Vec, mas para séries temporais |
| `Informer`, `Autoformer` | Temporal + Deep Learning | Forecasting com representação interna vetorial |

---

## 🧪 Exemplo prático: embeddings de uma série temporal com `TS2Vec`

### 🔧 Instalar dependências:

```bash
pip install ts2vec
```

> Se `ts2vec` não estiver disponível diretamente no PyPI, você pode encontrar o repositório [no GitHub oficial: yusugomori/TS2Vec](https://github.com/yusugomori/TS2Vec).

---

### 📉 Exemplo em Python:

```python
import numpy as np
from ts2vec import TS2Vec
import matplotlib.pyplot as plt

# Simulação de série temporal: 100 dias, 4 variáveis (ex: preço, volume, RSI, MACD)
np.random.seed(42)
serie_temporal = np.random.randn(100, 4)

# Treinamento do modelo
model = TS2Vec(input_dims=4)
model.fit([serie_temporal], verbose=True)

# Extração do embedding (por janela temporal ou série completa)
embedding = model.encode([serie_temporal])[0]

print("Shape do embedding:", embedding.shape)

# Visualização (opcional)
plt.plot(embedding[:, 0])
plt.title("Embedding (dimensão 0) da série temporal")
plt.xlabel("Tempo")
plt.ylabel("Valor do embedding")
plt.show()
```

---

### 🔍 Resultado:

* O embedding tem **dimensões \[tempo, embedding\_dim]** — você obtém uma representação vetorial para cada ponto da série.
* Pode usar isso para: clustering, comparação de ativos, detecção de padrões, regime switching etc.

---
Ótimo! Vamos criar um exemplo completo com dados **reais do mercado financeiro**, usando o `yfinance` para baixar os dados de uma ação (ex: PETR4.SA) e gerar embeddings com `TS2Vec`.

---

## 📈 Exemplo: Embeddings de série temporal da Petrobras (PETR4.SA)

### 🔧 Passo 1: Instalar dependências

```bash
pip install yfinance ts2vec matplotlib
```

> Se `ts2vec` não estiver no PyPI, clone do GitHub:

```bash
git clone https://github.com/yusugomori/TS2Vec.git
cd TS2Vec
pip install -e .
```

---

### 💻 Passo 2: Código completo em Python

```python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ts2vec import TS2Vec

# Baixar dados da Petrobras (PETR4.SA)
ticker = yf.Ticker("PETR4.SA")
df = ticker.history(start="2022-01-01", end="2023-12-31")

# Selecionar colunas relevantes (preço e volume)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Normalizar os dados (escala semelhante para o modelo)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Convertendo para formato esperado pelo TS2Vec
data_series = data_scaled.astype(np.float32)

# Treinar o modelo TS2Vec
model = TS2Vec(input_dims=data_series.shape[1])
model.fit([data_series], verbose=True)

# Gerar embeddings
embedding = model.encode([data_series])[0]  # Shape: [tempo, embedding_dim]

print("Shape do embedding:", embedding.shape)

# Visualizar a 1ª dimensão do embedding ao longo do tempo
plt.plot(embedding[:, 0])
plt.title("Embedding da PETR4 - Dimensão 0")
plt.xlabel("Dias")
plt.ylabel("Valor do embedding")
plt.grid(True)
plt.show()
```

---

### ✅ O que esse código faz:

* Baixa os dados da **ação PETR4.SA** entre 2022 e 2023.
* Usa `Open`, `High`, `Low`, `Close`, e `Volume` como entrada.
* Treina o modelo `TS2Vec` para aprender um **embedding da série temporal**.
* Plota uma dimensão do embedding como exemplo.

---

### 🔍 O que você pode fazer com esse embedding:

* **Clusterizar ativos** similares.
* **Detectar regimes de mercado** (alta, baixa, lateralização).
* **Comparar períodos diferentes da mesma ação** (similaridade temporal).
* Alimentar modelos de previsão ou aprendizado não supervisionado.

---

### 💻 Exemplo embedding de texto

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased')

embeddings = model.encode(df.text)

```

---



