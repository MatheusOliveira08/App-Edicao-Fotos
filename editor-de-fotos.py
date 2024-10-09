import tkinter as tk #fornece uma interface gráfica para interagir com o usuário
from tkinter import filedialog #abre o explorador de arquivos para selecionar
import cv2 #biblioteca para processamento de imagens (manipula imagens/vídeos)
from PIL import Image, ImageTk #abrir, manipular e processar imagens | Coverte imagens para um formato que o tkinter consegue exibir
import numpy as np

def carregar_imagem(flag = None):
    global img_carregada
    caminho_arquivo = filedialog.askopenfilename()
    if caminho_arquivo:
        if flag == "acinzentar":
            img_carregada = cv2.imread(caminho_arquivo, cv2.IMREAD_GRAYSCALE) #lê a imagem com um canal no caminho especificado e armazena
        else:
            img_carregada = cv2.imread(caminho_arquivo) #lê a imagem no caminho especificado e armazena

        print(f"Dimensões da imagem: {img_carregada.shape}")
        exibir_imagem(img_carregada, eh_original=True)
        limpar_tela()

def exibir_imagem(img_carregada, eh_original = False):
    img_rgb = cv2.cvtColor(img_carregada, cv2.COLOR_BGR2RGB) #converte de BRG (padrão do OpenCV) para RGB para poder mexer com o PIL e o tkinter

    #proporções da tela
    largura_maxima = 500
    altura_maxima = 550

    img_pil, x_offset, y_offset = centralizar_imagem(img_rgb, largura_maxima, altura_maxima)
    img_tk = ImageTk.PhotoImage(img_pil) #converte de Image para ImageTk (formato que o tkinter consegue exibir)

    if eh_original:
        tela_imagem_original.delete("all") #limpa a tela antes de adicionar a nova imagem
        tela_imagem_original.image = img_tk #define img_tk como um atributo da tela (mantém a referência) para não cair no garbage collector
        tela_imagem_original.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
    else:
        tela_imagem_editada.delete("all")
        tela_imagem_editada.image = img_tk
        tela_imagem_editada.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)

def centralizar_imagem(img_rgb, largura_maxima, altura_maxima):
    img_pil = Image.fromarray(img_rgb) #img_rbg é uma matriz, precisa converter para um objeto Image para poder manipular/converter
    img_pil.thumbnail((largura_maxima, altura_maxima)) #redimensiona a imagem para o caso de ser maior 500x550

    #calcula o deslocamento de x e y para centralizar a imagem na tela
    x_offset = (largura_maxima - img_pil.width) // 2
    y_offset = (altura_maxima - img_pil.height) // 2

    return img_pil, x_offset, y_offset #retorna a imagem redimensionada e os deslocamentos

def aplicar_filtros_pb(tipo_filtro):
    if img_carregada is None:
        return
    if tipo_filtro == 0:
        img_filtrada = cv2.blur(img_carregada, (15,15))
    elif tipo_filtro == 1:
        img_filtrada = cv2.medianBlur(img_carregada, 5)
    elif tipo_filtro == 2:
        img_filtrada = cv2.GaussianBlur(img_carregada, (5,5), 2)
    else:
        return
    
    exibir_imagem(img_filtrada)

def aplicar_filtros_pa(tipo_filtro):
    if img_carregada is None:
        return

    laplaciano_kernel = np.array([[-1, -1, -1], #8-conectividade (Diagonais Incluídas)
                                  [-1, 8, -1],
                                  [-1, -1, -1]])
    
    if tipo_filtro == 0:
        sobel_x = cv2.Sobel(img_carregada, cv2.CV_64F, 1, 0, ksize=3)
        img_filtrada = cv2.convertScaleAbs(sobel_x)
    elif tipo_filtro == 1:
        sobel_y = cv2.Sobel(img_carregada, cv2.CV_64F, 0, 1, ksize=3)
        img_filtrada = cv2.convertScaleAbs(sobel_y)
    elif tipo_filtro == 2:
        sobel_x = cv2.Sobel(img_carregada, cv2.CV_64F, 1, 0, ksize=3)  #direção horizontal
        sobel_y = cv2.Sobel(img_carregada, cv2.CV_64F, 0, 1, ksize=3)  #direção vertical
        img_filtrada = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0) #combina as bordas
    elif tipo_filtro == 3:
        img_filtrada = cv2.Canny(img_carregada, 50, 100) #descarta < threshold1, borda válida > threshold2, entre: só se conectado a bordas fortes
    elif tipo_filtro == 4:
        img_filtrada = cv2.filter2D(img_carregada, cv2.CV_64F, laplaciano_kernel) #CV_64F constante de ponto flutuante para evitar overflow no calculo das bordas
        img_filtrada = cv2.convertScaleAbs(img_filtrada) #converte para 8 bits para visualização 
    else:
        return

    exibir_imagem(img_filtrada)

def limpar_tela():
    tela_imagem_editada.delete("all")


#Cria a interface gráfica
aplicativo = tk.Tk()
aplicativo.title("Aplicativo de Processamento de Imagens")
aplicativo.geometry("1085x600") #tamanho da janela
aplicativo.config(bg="#2e2e2e") #cor do fundo

#Variável que vai receber a imagem que o usuário selecionar
img_carregada = None

#Cria o menu da aplicação 
menu_apk = tk.Menu(aplicativo)
aplicativo.config(menu = menu_apk)

#Menu do Arquivo
arquivo_menu = tk.Menu(menu_apk, tearoff=0)
menu_apk.add_cascade(label="Arquivo", menu=arquivo_menu)
arquivo_menu.add_command(label="Carregar Imagem", command=carregar_imagem)
arquivo_menu.add_separator()
arquivo_menu.add_command(label="Carregar Imagem Cinza", command=lambda: carregar_imagem("acinzentar"))
arquivo_menu.add_separator()
arquivo_menu.add_command(label="Sair", command=aplicativo.quit)

#Menu dos Filtros
filtros_menu = tk.Menu(menu_apk, tearoff=0)
menu_apk.add_cascade(label="Filtros", menu=filtros_menu)

#Submenu Filtros Passa-Baixa 
submenu_passa_baixa = tk.Menu(filtros_menu, tearoff=0)
submenu_passa_baixa.add_command(label="Filtro de Média", command=lambda:aplicar_filtros_pb(0))
submenu_passa_baixa.add_separator()
submenu_passa_baixa.add_command(label="Filtro de Mediana", command=lambda:aplicar_filtros_pb(1)) 
submenu_passa_baixa.add_separator()
submenu_passa_baixa.add_command(label="Filtro Gaussiano", command=lambda:aplicar_filtros_pb(2)) 

#Submenu Filtros Passa-Alta
submenu_passa_alta = tk.Menu(filtros_menu, tearoff=0)
submenu_passa_alta.add_command(label="Filtro Sobel Horizontal", command=lambda:aplicar_filtros_pa(0)) 
submenu_passa_alta.add_separator()
submenu_passa_alta.add_command(label="Filtro Sobel Vertical", command=lambda:aplicar_filtros_pa(1)) 
submenu_passa_alta.add_separator()
submenu_passa_alta.add_command(label="Filtro Sobel", command=lambda:aplicar_filtros_pa(2)) 
submenu_passa_alta.add_separator()
submenu_passa_alta.add_command(label="Filtro Canny", command=lambda:aplicar_filtros_pa(3)) 
submenu_passa_alta.add_separator()
submenu_passa_alta.add_command(label="Filtro Laplaciano", command=lambda:aplicar_filtros_pa(4))

# Adiciona os submenus ao menu principal Filtros
filtros_menu.add_cascade(label="Passa-Baixa", menu=submenu_passa_baixa)
filtros_menu.add_cascade(label="Passa-Alta", menu=submenu_passa_alta)

#Cria a tela onde fica a imagem original
tela_imagem_original = tk.Canvas(aplicativo, width=500, height=550, bg="#2e2e2e", highlightthickness=1, highlightbackground="gray")
tela_imagem_original.grid(row=0, column=0, padx=20, pady=20) #padx e pady define o espaçamento horizontal e vertical da tela

#Cria a tela onde fica a imagem editada
tela_imagem_editada = tk.Canvas(aplicativo, width=500, height=550, bg="#2e2e2e", highlightthickness=1, highlightbackground="gray")
tela_imagem_editada.grid(row=0, column=1, padx=20, pady=20) #aqui o column é 1, pois vão ficar duas telas uma do lado da outra (0x0 e 0x1)


aplicativo.mainloop()

