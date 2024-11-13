import numpy as np

def erodir(img_binaria, kernel):
    altura, largura = img_binaria.shape
    k_altura, k_largura = kernel.shape
    k_center_x, k_center_y = k_altura // 2, k_largura // 2

    # Criar uma cópia para a imagem erodida
    img_erodida = np.zeros_like(img_binaria)

    # Aplicar erosão
    for y in range(k_center_y, altura - k_center_y):
        for x in range(k_center_x, largura - k_center_x):
            # Verificar se todos os pixels dentro da área do kernel são brancos
            area = img_binaria[y - k_center_y:y + k_center_y + 1, x - k_center_x:x + k_center_x + 1]
            if np.all(area[kernel == 1] == 255):  # Todos os pixels no kernel devem ser 255
                img_erodida[y, x] = 255

    return img_erodida

def dilatar(img_binaria, kernel):
    altura, largura = img_binaria.shape
    k_altura, k_largura = kernel.shape
    k_center_x, k_center_y = k_altura // 2, k_largura // 2

    # Criar uma cópia para a imagem dilatada
    img_dilatada = np.zeros_like(img_binaria)

    # Aplicar dilatação
    for y in range(k_center_y, altura - k_center_y):
        for x in range(k_center_x, largura - k_center_x):
            # Verificar se algum pixel dentro da área do kernel é branco
            area = img_binaria[y - k_center_y:y + k_center_y + 1, x - k_center_x:x + k_center_x + 1]
            if np.any(area[kernel == 1] == 255):  # Qualquer pixel no kernel sendo 255 ativa a dilatação
                img_dilatada[y, x] = 255

    return img_dilatada