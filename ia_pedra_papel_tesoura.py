"""
Camada em português que esconde o TensorFlow e o OpenCV por trás de
funções simples: tirar_fotos(), treinar(), testar() e jogar().

Quem for usar este módulo não precisa entender de "modelos", "redes
neurais" ou "classes" — só precisa chamar essas quatro funções, nessa
ordem, dentro do notebook.
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import json
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

PASTA_DADOS = "data"
PASTA_MODELOS = "models"
JOGADAS = ["pedra", "papel", "tesoura"]
TAMANHO_IMAGEM = (128, 128)
ARQUIVO_MODELO = os.path.join(PASTA_MODELOS, "modelo_pedra_papel_tesoura.keras")
ARQUIVO_CLASSES = os.path.join(PASTA_MODELOS, "classes.json")
EMOJI_DA_JOGADA = {"pedra": "✊", "papel": "✋", "tesoura": "✌️"}


# ---------------------------------------------------------------------------
# Funções que as crianças usam no notebook
# ---------------------------------------------------------------------------

def tirar_fotos(jogada, quantidade=20):
    """Abre a webcam para tirar fotos de uma jogada (pedra, papel ou tesoura).

    Pressione ESPAÇO para tirar cada foto e ESC para parar antes.
    As fotos são salvas em data/<jogada>/.
    """
    if jogada not in JOGADAS:
        print(f"❌ '{jogada}' não é uma jogada válida. Use: {', '.join(JOGADAS)}.")
        return

    pasta = os.path.join(PASTA_DADOS, jogada)
    os.makedirs(pasta, exist_ok=True)
    numero_inicial = len(os.listdir(pasta))

    camera = _abrir_webcam()
    if camera is None:
        return

    print(f"📷 Faça a pose de '{jogada.upper()}' na frente da câmera.")
    print("Pressione ESPAÇO para tirar cada foto e ESC para parar antes.")

    fotos_tiradas = 0
    while fotos_tiradas < quantidade:
        ok, quadro = camera.read()
        if not ok:
            print("❌ Não consegui ler a imagem da webcam.")
            break

        quadro_para_mostrar = quadro.copy()
        texto = f"{jogada}: foto {fotos_tiradas + 1} de {quantidade}  (ESPACO=tirar, ESC=parar)"
        cv2.putText(quadro_para_mostrar, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Tirar fotos - Pedra, Papel ou Tesoura", quadro_para_mostrar)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:  # ESC
            break
        if tecla == 32:  # ESPACO
            nome_arquivo = f"foto_{numero_inicial + fotos_tiradas:04d}.jpg"
            cv2.imwrite(os.path.join(pasta, nome_arquivo), quadro)
            fotos_tiradas += 1

    camera.release()
    cv2.destroyAllWindows()
    print(f"✅ Foram salvas {fotos_tiradas} fotos novas de '{jogada}'.")


def treinar(quantidade_de_rodadas=10):
    """Treina o robô com as fotos que estão nas pastas data/pedra, data/papel
    e data/tesoura, e salva o resultado para ser usado por testar() e jogar().
    """
    _garantir_pastas()

    print("📸 Procurando as fotos que você tirou...")
    imagens, rotulos, nomes_das_classes = _carregar_fotos_e_rotulos()

    if len(nomes_das_classes) < 2:
        print("⚠️  Preciso de fotos de pelo menos 2 jogadas diferentes.")
        print('Use tirar_fotos("pedra"), tirar_fotos("papel") ou tirar_fotos("tesoura") primeiro!')
        return

    if len(imagens) < len(nomes_das_classes) * 5:
        print("⚠️  Você tem poucas fotos. Tire pelo menos 10 fotos de cada jogada para o robô aprender melhor.")

    print(f"✅ Encontrei {len(imagens)} fotos de {len(nomes_das_classes)} jogadas: {', '.join(nomes_das_classes)}")

    imagens_treino, rotulos_treino, imagens_validacao, rotulos_validacao = _separar_treino_e_validacao(
        imagens, rotulos
    )

    print("🧠 Montando o cérebro do robô...")
    modelo = _construir_rede_neural(len(nomes_das_classes))

    print("🏋️  Treinando... isso pode levar um tempinho.")
    historico = modelo.fit(
        imagens_treino,
        rotulos_treino,
        validation_data=(imagens_validacao, rotulos_validacao),
        epochs=quantidade_de_rodadas,
        verbose=1,
    )

    _salvar_modelo(modelo, nomes_das_classes)

    acerto_final = historico.history["val_accuracy"][-1] * 100
    print(f"\n🎉 Pronto! Na última rodada, o robô acertou {acerto_final:.0f}% das fotos novas.")

    _mostrar_grafico_de_aprendizado(historico)


def testar(caminho_da_imagem=None):
    """Mostra o que o robô acha de uma foto: pedra, papel ou tesoura.

    Se você não passar um caminho de imagem, a webcam abre para você
    tirar uma foto na hora.
    """
    modelo, nomes_das_classes = _carregar_modelo()
    if modelo is None:
        print("❌ Ainda não existe um robô treinado. Use treinar() primeiro!")
        return

    if caminho_da_imagem is None:
        imagem = _tirar_uma_foto_da_webcam("Mostre sua jogada e pressione ESPACO")
        if imagem is None:
            print("Nenhuma foto foi tirada.")
            return
    else:
        imagem = cv2.imread(caminho_da_imagem)
        if imagem is None:
            print(f"❌ Não encontrei a imagem em '{caminho_da_imagem}'.")
            return

    nome_da_jogada, confianca = _prever_jogada(modelo, nomes_das_classes, imagem)

    plt.figure(figsize=(4, 4))
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Eu acho que é {nome_da_jogada.upper()} {_emoji(nome_da_jogada)}\n({confianca:.0f}% de certeza)")
    plt.show()

    print(f"🤖 O robô acha que a imagem mostra: {nome_da_jogada.upper()} (confiança de {confianca:.0f}%)")


def jogar():
    """Joga uma rodada de pedra, papel e tesoura contra o computador,
    usando a webcam para ver a sua jogada.
    """
    modelo, nomes_das_classes = _carregar_modelo()
    if modelo is None:
        print("❌ Ainda não existe um robô treinado. Use treinar() primeiro!")
        return

    imagem = _tirar_uma_foto_da_webcam("Mostre PEDRA, PAPEL ou TESOURA e pressione ESPACO")
    if imagem is None:
        print("Nenhuma foto foi tirada. Vamos jogar outra hora!")
        return

    jogada_pessoa, confianca = _prever_jogada(modelo, nomes_das_classes, imagem)
    jogada_computador = random.choice(nomes_das_classes)
    resultado = _decidir_vencedor(jogada_pessoa, jogada_computador)

    plt.figure(figsize=(4, 4))
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Você jogou {jogada_pessoa.upper()} {_emoji(jogada_pessoa)}")
    plt.show()

    print(f"🧑 Você jogou: {jogada_pessoa.upper()} {_emoji(jogada_pessoa)}  (o robô tinha {confianca:.0f}% de certeza)")
    print(f"🤖 O computador jogou: {jogada_computador.upper()} {_emoji(jogada_computador)}")

    if resultado == "empate":
        print("🤝 Empate! Joguem de novo!")
    elif resultado == "pessoa":
        print("🎉 Você ganhou!")
    else:
        print("😅 O computador ganhou! Tente de novo!")


# ---------------------------------------------------------------------------
# Funções internas (o "motor" escondido por trás do TensorFlow/OpenCV)
# ---------------------------------------------------------------------------

def _garantir_pastas():
    os.makedirs(PASTA_DADOS, exist_ok=True)
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    for jogada in JOGADAS:
        os.makedirs(os.path.join(PASTA_DADOS, jogada), exist_ok=True)


def _abrir_webcam():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
    if not camera.isOpened():
        print("❌ Não consegui abrir a webcam. Verifique se ela está conectada e livre.")
        return None
    return camera


def _tirar_uma_foto_da_webcam(mensagem):
    camera = _abrir_webcam()
    if camera is None:
        return None

    foto_capturada = None
    while True:
        ok, quadro = camera.read()
        if not ok:
            break

        quadro_para_mostrar = quadro.copy()
        cv2.putText(quadro_para_mostrar, mensagem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Sorria para a camera!", quadro_para_mostrar)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 32:  # ESPACO
            foto_capturada = quadro
            break
        if tecla == 27:  # ESC
            break

    camera.release()
    cv2.destroyAllWindows()
    return foto_capturada


def _pre_processar_imagem(imagem_bgr):
    imagem_redimensionada = cv2.resize(imagem_bgr, TAMANHO_IMAGEM)
    imagem_rgb = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2RGB)
    return imagem_rgb.astype("float32") / 255.0


def _carregar_fotos_e_rotulos():
    imagens = []
    rotulos = []
    nomes_das_classes = []

    for jogada in JOGADAS:
        pasta = os.path.join(PASTA_DADOS, jogada)
        if not os.path.isdir(pasta):
            continue
        arquivos = [f for f in os.listdir(pasta) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not arquivos:
            continue

        indice_da_classe = len(nomes_das_classes)
        nomes_das_classes.append(jogada)

        for nome_arquivo in arquivos:
            imagem = cv2.imread(os.path.join(pasta, nome_arquivo))
            if imagem is None:
                continue
            imagens.append(_pre_processar_imagem(imagem))
            rotulos.append(indice_da_classe)

    return np.array(imagens), np.array(rotulos), nomes_das_classes


def _separar_treino_e_validacao(imagens, rotulos):
    indices = np.arange(len(imagens))
    np.random.shuffle(indices)
    imagens, rotulos = imagens[indices], rotulos[indices]

    corte = max(1, int(len(imagens) * 0.8))
    imagens_treino, imagens_validacao = imagens[:corte], imagens[corte:]
    rotulos_treino, rotulos_validacao = rotulos[:corte], rotulos[corte:]

    if len(imagens_validacao) == 0:
        imagens_validacao, rotulos_validacao = imagens_treino, rotulos_treino

    return imagens_treino, rotulos_treino, imagens_validacao, rotulos_validacao


def _construir_rede_neural(numero_de_classes):
    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=TAMANHO_IMAGEM + (3,),
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False
        modelo = tf.keras.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(numero_de_classes, activation="softmax"),
        ])
    except Exception:
        print("⚠️  Não consegui baixar uma rede neural pronta da internet. Vou criar uma do zero.")
        modelo = tf.keras.Sequential([
            tf.keras.layers.Input(shape=TAMANHO_IMAGEM + (3,)),
            tf.keras.layers.Conv2D(16, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(numero_de_classes, activation="softmax"),
        ])

    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def _salvar_modelo(modelo, nomes_das_classes):
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    modelo.save(ARQUIVO_MODELO)
    with open(ARQUIVO_CLASSES, "w", encoding="utf-8") as arquivo:
        json.dump(nomes_das_classes, arquivo, ensure_ascii=False)


def _carregar_modelo():
    if not os.path.exists(ARQUIVO_MODELO) or not os.path.exists(ARQUIVO_CLASSES):
        return None, None
    modelo = tf.keras.models.load_model(ARQUIVO_MODELO)
    with open(ARQUIVO_CLASSES, "r", encoding="utf-8") as arquivo:
        nomes_das_classes = json.load(arquivo)
    return modelo, nomes_das_classes


def _prever_jogada(modelo, nomes_das_classes, imagem_bgr):
    imagem_pronta = _pre_processar_imagem(imagem_bgr)
    previsao = modelo.predict(np.expand_dims(imagem_pronta, axis=0), verbose=0)[0]
    indice_da_classe = int(np.argmax(previsao))
    confianca = float(previsao[indice_da_classe]) * 100
    return nomes_das_classes[indice_da_classe], confianca


def _decidir_vencedor(jogada_pessoa, jogada_computador):
    if jogada_pessoa == jogada_computador:
        return "empate"
    vence_contra = {"pedra": "tesoura", "papel": "pedra", "tesoura": "papel"}
    if vence_contra.get(jogada_pessoa) == jogada_computador:
        return "pessoa"
    return "computador"


def _emoji(jogada):
    return EMOJI_DA_JOGADA.get(jogada, "")


def _mostrar_grafico_de_aprendizado(historico):
    plt.figure(figsize=(6, 4))
    plt.plot(historico.history["accuracy"], label="Acertos no treino", marker="o")
    plt.plot(historico.history["val_accuracy"], label="Acertos em fotos novas", marker="o")
    plt.title("Como o robô foi aprendendo")
    plt.xlabel("Rodada de treino")
    plt.ylabel("Taxa de acerto")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
