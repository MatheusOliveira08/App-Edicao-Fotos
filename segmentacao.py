import numpy as np

def calcular_limiar_global(img, tolerancia=0.5):
    T = np.mean(img)  # Estimativa inicial
    while True:
        G1 = img[img > T]
        G2 = img[img <= T]

        # Calcular a média dos tons de cinza de G1 e G2
        m1 = np.mean(G1) if len(G1) > 0 else 0
        m2 = np.mean(G2) if len(G2) > 0 else 0

        # Calcular o novo limiar
        novo_T = (m1 + m2) / 2

        # Verificar se a diferença é menor que a tolerância
        if abs(T - novo_T) < tolerancia:
            break

        T = novo_T

    return T

# Função para aplicar a limiarização binária
def limiarizacao_binaria(img, limiar):
    img_binaria = np.zeros_like(img, dtype=np.uint8)
    img_binaria[img > limiar] = 255
    return img_binaria

def limiarizacao_adaptativa_otsu(img, tamanho_janela, c=0):
    # Calcular margens para o padding com base no tamanho da janela
    margem = tamanho_janela // 2
    
    # Adicionar reflexão nas bordas usando np.pad
    img_refletida = np.pad(img, ((margem, margem), (margem, margem)), mode='reflect')
    
    # Obter dimensões da imagem original
    altura, largura = img.shape
    
    # Criar uma imagem binária vazia para o resultado
    img_binaria = np.zeros((altura, largura), dtype=np.uint8)

    # Iterar sobre cada pixel da imagem original
    for y in range(altura):
        for x in range(largura):
            # Extrair a janela ao redor do pixel atual da imagem
            janela = img_refletida[y:y + tamanho_janela, x:x + tamanho_janela]
            
            # Calcular a média da janela e ajustar com o valor constante c
            media_janela = np.mean(janela)
            limiar_local = media_janela - c
            
            # Aplicar o limiar para definir o pixel na imagem binária
            if img[y, x] > limiar_local:
                img_binaria[y, x] = 255
            else:
                img_binaria[y, x] = 0

    return img_binaria