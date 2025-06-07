
# Projeto de Classificação de Áudio com Redes Neurais

Este projeto implementa diferentes modelos de redes neurais para classificação de áudios, com foco em reconhecimento e análise de características sonoras. Os modelos implementados incluem MLP, CNN, RNN, LSTM e GRU, e são avaliados por métricas padrão de aprendizado de máquina.

Os script foram executados utilizando [ROCm](https://dl.acm.org/doi/10.1145/3658644.3670285)

Apesar de parecer que está buscando "CUDA", o PyTorch com ROCm mapeia internamente isso para a GPU AMD corretamente, desde que:
- O PyTorch tenha sido instalado com suporte a ROCm e você esteja com `HIP_VISIBLE_DEVICES` e drivers ROCm corretamente configurados.

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7)
``` 

## Estrutura do Projeto

```
/audio-classification-project
├── models/                        # Código dos modelos de redes neurais
│   ├── mlp.py
│   ├── cnn.py
│   ├── rnn.py
│   ├── lstm.py
│   └── gru.py
├── view/                          # Código que gera os gráficos de análise
│   └── metrics.py
├── dataset/                       # Pasta para armazenar e carregar os arquivos que serão análisados
│   └── loadaudiodataset.py
└── README.md                      # Este arquivo
└── requirements.md                # Arquivo de depêndencias
```

## Modelos Implementados

- **MLP (Multilayer Perceptron)**: Rede neural totalmente conectada.
- **CNN (Convolutional Neural Network)**: Rede com camadas convolucionais para extração automática de características.
- **RNN (Recurrent Neural Network)**: Rede com memória para sequências temporais.
- **LSTM (Long Short-Term Memory)**: Variante de RNN que evita o problema do gradiente desaparecendo.
- **GRU (Gated Recurrent Unit)**: Variante simplificada do LSTM com desempenho similar.

## Métricas de Avaliação

As métricas implementadas são:

- **Acurácia**
- **Precisão**
- **Recall**
- **F1-Score**
- **Tempo de Inferência**

## Bibliotecas Utilizadas

- `numpy`: operações numéricas.
- `pandas`: manipulação de dados tabulares.
- `torch` (PyTorch): framework para construção e treinamento das redes neurais.
- `scikit-learn`: métricas e pré-processamento.
- `matplotlib` / `seaborn`: visualização de resultados.
- `librosa`: processamento e extração de características de áudio.

## Como Instalar as Dependências

Recomenda-se criar um ambiente virtual e instalar as dependências com:

```bash
pip install -r requirements.txt
```

Conteúdo básico do `requirements.txt`:

```
numpy
pandas
torch
scikit-learn
matplotlib
seaborn
librosa
```

## Como Rodar o Projeto

1. Prepare seu dataset de áudio e coloque na pasta `data/`.
2. Treine o modelo desejado executando.

Você pode utilizar os arquivos .py seguindo os nomes, como `mlp` ou os outros modelos: `cnn`, `rnn`, `lstm` ou `gru`.

3. Para visualizar análises, utilize os notebooks na pasta `views/metrics.py`.

## Datasets Utilizados

|  Tipos de Ataques  | Anos | Dataset   |  Número de Áudios  <br>（Subdataset：Real/Fake） 	  |    Lingua   |
|:-----------:|:------------:|:------------:|:-------------:|:------------:|
|Vocoder|2024|CVoiceFake<br>[Paper](https://dl.acm.org/doi/10.1145/3658644.3670285) [Dataset](https://safeearweb.github.io/Project/)|23,544/91,700|Multi-lingual|
|Vocoder|2024|MLADDC<br>[Paper](https://openreview.net/forum?id=ic3HvoOTeU) [Dataset](https://speech007.github.io/MLADDC_Nips/)|80k/160k|Multi-lingual|
|VC, Replay<br>and Adversarial|2024|VSASV<br>[Paper](https://www.isca-archive.org/interspeech_2024/hoang24b_interspeech.html) [Dataset](https://github.com/hustep-lab/VSASV-Dataset)|164,000/174,000|Multi-lingual|