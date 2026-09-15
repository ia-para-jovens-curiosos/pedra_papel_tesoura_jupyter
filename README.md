# 🎮 IA para Jovens Curiosos — Pedra, Papel e Tesoura

[![GitHub Repo](https://img.shields.io/badge/GitHub-ia--para--jovens--curiosos%2Fpedra__papel__tesoura__jupyter-blue?logo=github)](https://github.com/ia-para-jovens-curiosos/pedra_papel_tesoura_jupyter)

Um projeto para crianças treinarem sua própria Inteligência Artificial, usando a webcam, para jogar pedra, papel e tesoura contra o computador.

## Como abrir

Abra o arquivo `sample.ipynb` no Jupyter/PyCharm e execute as células de cima para baixo, com `Shift + Enter`.

O notebook guia você por quatro passos:

1. **Tirar fotos** — tirar fotos da sua mão fazendo pedra, papel e tesoura
2. **Treinar** — ensinar o robô a reconhecer cada jogada
3. **Testar** — conferir se o robô aprendeu direito
4. **Jogar** — jogar pedra, papel e tesoura contra o computador

## Como funciona por baixo dos panos

Todo o TensorFlow e o OpenCV ficam escondidos dentro do arquivo `ia_pedra_papel_tesoura.py`, que oferece só quatro funções simples em português:

- `tirar_fotos(jogada, quantidade)`
- `treinar()`
- `testar(caminho_da_imagem=None)`
- `jogar()`

As fotos tiradas ficam salvas em `data/pedra`, `data/papel` e `data/tesoura`, e o robô treinado é salvo em `models/`.

## Ambiente

Este projeto usa **Python 3.12** (o TensorFlow ainda não funciona em versões mais novas do Python). O ambiente virtual já está configurado em `.venv`. Para recriá-lo do zero:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

É necessário ter uma webcam conectada para tirar fotos, testar e jogar.
