import numpy as np

def aplicar_convolucao(img, mascara, tipo_filtro):
    # Dimensões da imagem e da máscara
    altura, largura = img.shape
    m_altura, m_largura = mascara.shape

    # Calcula a margem da máscara (para centralizar)
    margem_y = m_altura // 2
    margem_x = m_largura // 2

    # Adiciona padding na imagem com reflexão nas bordas
    img_bordada = np.pad(img, ((margem_y, margem_y), (margem_x, margem_x)), mode='reflect')

    # Cria uma imagem de saída vazia com a mesma forma
    img_filtrada = np.zeros_like(img)

    # Aplica a convolução
    for i in range(altura):
        for j in range(largura):
            # Extrai a região da imagem que corresponde ao kernel
            regiao = img_bordada[i:i + m_altura, j:j + m_largura]

            # Calcula a convolução (multiplicação elemento a elemento e soma)
            if tipo_filtro in ["media", "gauss", "sobel"]:
                valor = np.sum(regiao * mascara)
            elif tipo_filtro == "mediana":
                valor = np.median(regiao)
            else:
                raise ValueError("Tipo de filtro não reconhecido. Use 'media' ou 'mediana'.")

            img_filtrada[i, j] = min(max(int(valor), 0), 255)  # Limita entre 0 e 255

    return img_filtrada

def criar_mascara_media(tamanho=5):
    # Certifique-se de que o tamanho seja ímpar
    if tamanho % 2 == 0:
        raise ValueError("O tamanho da máscara deve ser ímpar.")

    # Cria uma máscara de média (todos os elementos com o mesmo valor)
    valor = 1 / (tamanho * tamanho)
    return np.full((tamanho, tamanho), valor)

def criar_mascara_gaussiana(tamanho=5, sigma=1):
    if tamanho % 2 == 0:
        raise ValueError("O tamanho da máscara deve ser ímpar.")
    
    # Coordenadas para o centro
    ax = np.arange(-tamanho // 2 + 1, tamanho // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    
    # Fórmula Gaussiana
    mascara = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    
    # Normaliza para que a soma dos elementos seja 1
    mascara /= np.sum(mascara)
    return mascara

def criar_mascara_sobel(tamanho, direcao="horizontal"):
    if tamanho % 2 == 0 or tamanho < 3:
        raise ValueError("O tamanho da máscara Sobel deve ser um número ímpar maior ou igual a 3.")

    # Inicializar a máscara com zeros
    mascara = np.zeros((tamanho, tamanho), dtype=np.float32)
    meio = tamanho // 2

    # Define os valores de peso proporcional a máscara Sobel 3x3
    for i in range(tamanho):
        for j in range(tamanho):
            # Calcula a distância ao centro da máscara
            dy = i - meio
            dx = j - meio

            # Atribui os valores de acordo com a direção
            if direcao == "horizontal":
                mascara[i, j] = dx * (meio - abs(dy))
            elif direcao == "vertical":
                mascara[i, j] = dy * (meio - abs(dx))
            else:
                raise ValueError("Direção não reconhecida. Use 'horizontal' ou 'vertical'.")

    # Normaliza a máscara para manter a proporção da máscara 3x3
    fator = meio
    mascara /= fator

    return mascara

def criar_mascara_laplaciana(tipo="padrao"):
    if tipo == "padrao":
        # Máscara padrão 3x3
        return np.array([[0, -1, 0],
                         [-1, 4, -1],
                         [0, -1, 0]], dtype=np.float32)
    elif tipo == "intensa":
        # Máscara mais intensa 3x3
        return np.array([[-1, -1, -1],
                         [-1, 8, -1],
                         [-1, -1, -1]], dtype=np.float32)
    else:
        raise ValueError("Tipo de máscara não reconhecido. Use 'padrao' ou 'intensa'.")

def calcular_gradiente(img):
    # Aplica o filtro Sobel para calcular o gradiente em x e y
    sobel_horizontal = criar_mascara_sobel(3, "horizontal")
    sobel_vertical = criar_mascara_sobel(3, "vertical")
    
    grad_x = aplicar_convolucao(img, sobel_horizontal, tipo_filtro="media")
    grad_y = aplicar_convolucao(img, sobel_vertical, tipo_filtro="media")
    
    # Calcula a magnitude e a direção do gradiente
    magnitude = np.sqrt(grad_x**2 + grad_y**2) * 8
    direcao = np.arctan2(grad_y, grad_x)
    
    # Normaliza a magnitude para a faixa de 0 a 255
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude, direcao

def supressao_nao_maximos(magnitude, direcao):
    # Converte a direção do gradiente para ângulos em graus
    direcao = np.rad2deg(direcao) % 180

    altura, largura = magnitude.shape
    resultado = np.zeros((altura, largura), dtype=np.uint8)

    for i in range(1, altura - 1):
        for j in range(1, largura - 1):
            # Determina a direção da borda
            angulo = direcao[i, j]
            vizinho1, vizinho2 = 0, 0
            
            # Define os vizinhos a serem comparados com base na direção
            if (0 <= angulo < 22.5) or (157.5 <= angulo <= 180):
                vizinho1 = magnitude[i, j - 1]
                vizinho2 = magnitude[i, j + 1]
            elif 22.5 <= angulo < 67.5:
                vizinho1 = magnitude[i - 1, j + 1]
                vizinho2 = magnitude[i + 1, j - 1]
            elif 67.5 <= angulo < 112.5:
                vizinho1 = magnitude[i - 1, j]
                vizinho2 = magnitude[i + 1, j]
            elif 112.5 <= angulo < 157.5:
                vizinho1 = magnitude[i - 1, j - 1]
                vizinho2 = magnitude[i + 1, j + 1]

            # Suprime o valor se não for um máximo local
            if magnitude[i, j] >= vizinho1 and magnitude[i, j] >= vizinho2:
                resultado[i, j] = magnitude[i, j]

    return resultado

def aplicacao_histerese(img, limiar_baixo, limiar_alto):
    # Identifica bordas fortes e fracas com base nos limiares
    borda_forte = 255
    borda_fraca = 50

    resultado = np.zeros_like(img, dtype=np.uint8)

    bordas_fortes = (img >= limiar_alto)
    bordas_fracas = ((img >= limiar_baixo) & (img < limiar_alto))

    resultado[bordas_fortes] = borda_forte
    resultado[bordas_fracas] = borda_fraca

    # Realiza uma verificação de conectividade para conectar bordas fracas
    altura, largura = img.shape
    for i in range(1, altura - 1):
        for j in range(1, largura - 1):
            if resultado[i, j] == borda_fraca:
                # Conecta borda fraca a borda forte se for vizinha
                if np.any(resultado[i-1:i+2, j-1:j+2] == borda_forte):
                    resultado[i, j] = borda_forte
                else:
                    resultado[i, j] = 0

    return resultado

def filtro_media(img, tamanho_mascara): 
    mascara = criar_mascara_media(tamanho_mascara)
    return aplicar_convolucao(img, mascara, "media")

def filtro_mediana(img, tamanho_mascara): 
    mascara = criar_mascara_media(tamanho_mascara)
    return aplicar_convolucao(img, mascara, "mediana")

def filtro_gauss(img, tamanho_mascara, sigma):
    mascara = criar_mascara_gaussiana(tamanho_mascara, sigma)
    return aplicar_convolucao(img, mascara, "gauss")

def filtro_sobel(img, tamanho_mascara):
    # Criar máscaras Sobel para as direções horizontal e vertical
    mascara_sobel_horizontal = criar_mascara_sobel(tamanho_mascara, direcao="horizontal")
    mascara_sobel_vertical = criar_mascara_sobel(tamanho_mascara, direcao="vertical")
    
    # Aplicar Sobel horizontal e vertical
    sobel_horizontal = aplicar_convolucao(img, mascara_sobel_horizontal, tipo_filtro="sobel")
    sobel_vertical = aplicar_convolucao(img, mascara_sobel_vertical, tipo_filtro="sobel")
    
    # Calcular a magnitude do gradiente combinando as direções
    img_sobel = np.sqrt(sobel_horizontal**2 + sobel_vertical**2) * 8
    img_sobel = np.clip(img_sobel, 0, 255).astype(np.uint8)
    
    return img_sobel

def filtro_canny(img, limiar_baixo=50, limiar_alto=100):
    #Suavização com filtro Gaussiano
    img_suave = filtro_gauss(img, 5, 1)
    
    #Cálculo do gradiente
    magnitude, direcao = calcular_gradiente(img_suave)
    print(f"Magnitude: {magnitude}")
    #Supressão de não-máximos
    img_suprimida = supressao_nao_maximos(magnitude, direcao)
    
    #Aplicação de histerese
    img_canny = aplicacao_histerese(img_suprimida, limiar_baixo, limiar_alto)
    
    return img_canny

def filtro_laplaciano(img, tipo="padrao"):
    #Suavização com filtro Gaussiano
    img_suave = filtro_gauss(img, 5, 1)

    mascara_laplaciana = criar_mascara_laplaciana(tipo)
    
    img_laplaciana = aplicar_convolucao(img_suave, mascara_laplaciana, tipo_filtro="media")
    
    # Normaliza a imagem para a faixa de 0 a 255
    img_laplaciana = np.clip(img_laplaciana, 0, 255).astype(np.uint8)
    
    return img_laplaciana