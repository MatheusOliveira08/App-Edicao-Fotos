import tkinter as tk #fornece uma interface gráfica para interagir com o usuário
from tkinter import filedialog #abre o explorador de arquivos para selecionar
import cv2 #biblioteca para processamento de imagens (manipula imagens/vídeos)
from PIL import Image, ImageTk #abrir, manipular e processar imagens | Coverte imagens para um formato que o tkinter consegue exibir
import numpy as np

def carregar_imagem(flag = None):
    global img_carregada, img_processada, img_temp
    caminho_arquivo = filedialog.askopenfilename()
    if caminho_arquivo:
        if flag == "acinzentar":
            img_carregada = cv2.imread(caminho_arquivo, cv2.IMREAD_GRAYSCALE) #lê a imagem com um canal no caminho especificado e armazena
        else:
            img_carregada = cv2.imread(caminho_arquivo) #lê a imagem no caminho especificado e armazena

        img_processada = img_carregada.copy()
        img_temp = img_carregada.copy() 

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

def aplicar_filtros_pb(tipo_filtro, tamanho_kernel, sigma = 0):
    global img_processada, img_temp
    if img_temp is None:
        print("Nenhuma imagem carregada ou processada.")
        return
    
    if tamanho_kernel % 2 == 0:
        tamanho_kernel += 1  #Kernel sempre ímpar

    if tipo_filtro == 0:
        img_temp = cv2.blur(img_processada, (tamanho_kernel, tamanho_kernel))
    elif tipo_filtro == 1:
        img_temp = cv2.medianBlur(img_processada, tamanho_kernel)
    elif tipo_filtro == 2:
        img_temp = cv2.GaussianBlur(img_processada, (tamanho_kernel, tamanho_kernel), sigma)
    else:
        return
    
    exibir_imagem(img_temp)

def atualizar_filtro_passa_baixa(tipo_filtro):
    tamanho_kernel = slider_kernel_pb.get()  # Obtém o valor atual do slider
    sigma = slider_sigma.get() if tipo_filtro == 2 else 0
    aplicar_filtros_pb(tipo_filtro, tamanho_kernel, sigma)

def selecionar_filtro_pb(tipo_filtro):
    global filtro_selecionado
    filtro_selecionado = tipo_filtro

    frame_canny.grid_remove()
    frame_sigma.grid_remove()
    frame_slider_kernel_pa.grid_remove()
    frame_adaptativa.grid_remove()
    frame_binaria.grid_remove()
    frame_erodir.grid_remove()
    frame_dilatar.grid_remove()
    frame_slider_kernel_pb.grid(row=1, column=0, columnspan=2, pady=10)
    
    if tipo_filtro == 2: # Exibir o slider de sigma para o filtro Gaussiano
        frame_sigma.grid(row=2, column=0, columnspan=2, pady=5)
    else: # Ocultar o slider de sigma para outros filtros
        frame_sigma.grid_remove()

    confirma_filtro()

def aplicar_filtros_pa(tipo_filtro, tamanho_kernel, threshold1, threshold2, sobel_eixo):
    global img_processada, img_temp
    if img_temp is None:
        print("Nenhuma imagem carregada ou processada.")
        return
    
    if tamanho_kernel % 2 == 0:
        tamanho_kernel += 1  #Kernel sempre ímpar

    if tipo_filtro == 0:  # Filtro Sobel
        sobel_x = sobel_y = None
        if sobel_eixo in ["Horizontal", "Ambos"]:
            sobel_x = cv2.Sobel(img_processada, cv2.CV_64F, 1, 0, ksize = tamanho_kernel)  # Horizontal
        if sobel_eixo in ["Vertical", "Ambos"]:
            sobel_y = cv2.Sobel(img_processada, cv2.CV_64F, 0, 1, ksize = tamanho_kernel)  # Vertical

        if sobel_x is not None and sobel_y is not None:
            img_temp = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5,
                                       cv2.convertScaleAbs(sobel_y), 0.5, 0)
        if sobel_x is not None and sobel_y is None:
            img_temp = cv2.convertScaleAbs(sobel_x)  # Apenas Horizontal
        elif sobel_y is not None and sobel_x is None:
            img_temp = cv2.convertScaleAbs(sobel_y)  # Apenas Vertical
        elif sobel_x is not None and sobel_y is not None:
            img_temp = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5,
                                    cv2.convertScaleAbs(sobel_y), 0.5, 0)
    
    elif tipo_filtro == 1:  # Filtro Canny
        img_temp = cv2.Canny(img_processada, threshold1, threshold2)
    
    elif tipo_filtro == 2:  # Filtro Laplaciano
        img_temp = cv2.Laplacian(img_processada, cv2.CV_64F, ksize = tamanho_kernel)
        img_temp = cv2.convertScaleAbs(img_temp)
    
    exibir_imagem(img_temp)

def atualizar_filtro_passa_alta(tipo_filtro):
    tamanho_kernel = slider_kernel_pa.get()
    threshold1 = slider_canny1.get() if tipo_filtro == 1 else 0
    threshold2 = slider_canny2.get() if tipo_filtro == 1 else 0
    sobel_eixo = sobel_var.get() if tipo_filtro == 0 else ''
    aplicar_filtros_pa(tipo_filtro, tamanho_kernel, threshold1, threshold2, sobel_eixo)

def selecionar_filtro_pa(tipo_filtro):
    global filtro_selecionado
    filtro_selecionado = tipo_filtro
    
    frame_slider_kernel_pb.grid_remove()
    frame_adaptativa.grid_remove()
    frame_binaria.grid_remove()
    frame_erodir.grid_remove()
    frame_dilatar.grid_remove()
    frame_slider_kernel_pa.grid(row=1, column=0, columnspan=2, pady=5)
    
    if tipo_filtro == 0: # Exibir o slider da direção do filtro Sobel
        frame_sigma.grid_remove()
        frame_canny.grid_remove()
        frame_sobel.grid(row=2, column=0, columnspan=2, pady=5)
    elif tipo_filtro == 1: # Exibir o slider dos thresholds para o filtro de Canny
        frame_sigma.grid_remove()
        frame_sobel.grid_remove()
        frame_slider_kernel_pa.grid_remove()
        frame_canny.grid(row=2, column=0, columnspan=2, pady=5)
    else: # Ocultar o slider de sigma para outros filtros
        frame_sigma.grid_remove()
        frame_sobel.grid_remove()
        frame_canny.grid_remove()

    confirma_filtro()

def confirma_filtro():
    global img_processada, img_temp
    if img_temp is None:
        print("Nenhuma imagem carregada ou processada.")
        return

    # Confirma a imagem temporária como a nova imagem processada
    img_processada = img_temp.copy()
    exibir_imagem(img_processada)

def aplicar_limiarizacao_binaria(thresh, maxval=255):
    global img_processada, img_temp
    if img_temp is None:
        print("Nenhuma imagem carregada ou processada.")
        return
    
    # Limiarização Binária
    _, img_temp = cv2.threshold(img_processada, thresh, maxval, cv2.THRESH_BINARY)
    exibir_imagem(img_temp)

def aplicar_limiarizacao_adaptativa(blockSize, C, maxval=255):
    global img_processada, img_temp
    if img_temp is None:
        print("Nenhuma imagem carregada ou processada.")
        return
    
    if blockSize % 2 == 0:
        blockSize += 1  # blockSize deve ser ímpar

    # Limiarização Adaptativa
    img_temp = cv2.adaptiveThreshold(img_processada, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, blockSize, C)
    exibir_imagem(img_temp)

def exibir_sliders_binaria():
    frame_slider_kernel_pa.grid_remove()
    frame_slider_kernel_pb.grid_remove()
    frame_sigma.grid_remove()
    frame_sobel.grid_remove()
    frame_canny.grid_remove()
    frame_adaptativa.grid_remove()
    frame_erodir.grid_remove()
    frame_dilatar.grid_remove()
    frame_binaria.grid(row=2, column=0, columnspan=2, pady=5)

def exibir_sliders_adaptativa():
    frame_slider_kernel_pa.grid_remove()
    frame_slider_kernel_pb.grid_remove()
    frame_sigma.grid_remove()
    frame_sobel.grid_remove()
    frame_canny.grid_remove()
    frame_binaria.grid_remove()
    frame_erodir.grid_remove()
    frame_dilatar.grid_remove()
    frame_adaptativa.grid(row=2, column=0, columnspan=2, pady=5)

def aplicar_erodir(tamanho_kernel):
    global img_processada, img_temp
    if img_temp is None:
        print("Nenhuma imagem carregada ou processada.")
        return

    if tamanho_kernel % 2 == 0:
        tamanho_kernel += 1  # Kernel deve ser ímpar

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tamanho_kernel, tamanho_kernel))
    img_temp = cv2.erode(img_processada, kernel, iterations=1)
    exibir_imagem(img_temp)

def aplicar_dilatar(tamanho_kernel):
    global img_processada, img_temp
    if img_temp is None:
        print("Nenhuma imagem carregada ou processada.")
        return

    if tamanho_kernel % 2 == 0:
        tamanho_kernel += 1  # Kernel deve ser ímpar

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tamanho_kernel, tamanho_kernel))
    img_temp = cv2.dilate(img_processada, kernel, iterations=1)
    exibir_imagem(img_temp)

def exibir_sliders_erodir():
    frame_slider_kernel_pb.grid_remove()
    frame_slider_kernel_pa.grid_remove()
    frame_sigma.grid_remove()
    frame_sobel.grid_remove()
    frame_canny.grid_remove()
    frame_binaria.grid_remove()
    frame_adaptativa.grid_remove()
    frame_dilatar.grid_remove()
    frame_erodir.grid(row=2, column=0, columnspan=2, pady=5)

def exibir_sliders_dilatar():
    frame_slider_kernel_pb.grid_remove()
    frame_slider_kernel_pa.grid_remove()
    frame_sigma.grid_remove()
    frame_sobel.grid_remove()
    frame_canny.grid_remove()
    frame_binaria.grid_remove()
    frame_adaptativa.grid_remove()
    frame_erodir.grid_remove()
    frame_dilatar.grid(row=2, column=0, columnspan=2, pady=5)

def resetar():
    global img_carregada, img_processada, img_temp

    if img_carregada is None:
        print("Nenhuma imagem carregada.")
        return
    
    img_processada = img_carregada.copy()
    img_temp = img_carregada.copy()
    exibir_imagem(img_carregada)

def limpar_tela():
    tela_imagem_editada.delete("all")


#Cria a interface gráfica
aplicativo = tk.Tk()
aplicativo.title("Aplicativo de Processamento de Imagens")
aplicativo.geometry("1085x750") #tamanho da janela
aplicativo.config(bg="#2e2e2e") #cor do fundo

#Variáveis globais para as imagens
img_carregada = None #Imagem original
img_processada = None #Imagem para acumular os efeitos
img_temp = None #Imagem para ajustes com sliders

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

#Botão para resetar a imagem editada
reset = tk.Menu(menu_apk, tearoff=0)
menu_apk.add_command(label="Resetar", command=lambda: resetar())

#Menu dos Filtros
filtros_menu = tk.Menu(menu_apk, tearoff=0)
menu_apk.add_cascade(label="Filtros", menu=filtros_menu)

#Submenu Filtros Passa-Baixa 
submenu_passa_baixa = tk.Menu(filtros_menu, tearoff=0)
submenu_passa_baixa.add_command(label="Filtro de Média", command=lambda: selecionar_filtro_pb(0))
submenu_passa_baixa.add_separator()
submenu_passa_baixa.add_command(label="Filtro de Mediana", command=lambda: selecionar_filtro_pb(1))
submenu_passa_baixa.add_separator()
submenu_passa_baixa.add_command(label="Filtro Gaussiano", command=lambda: selecionar_filtro_pb(2))

#Submenu Filtros Passa-Alta
submenu_passa_alta = tk.Menu(filtros_menu, tearoff=0)
submenu_passa_alta.add_command(label="Filtro Sobel", command=lambda:selecionar_filtro_pa(0)) 
submenu_passa_alta.add_separator()
submenu_passa_alta.add_command(label="Filtro Canny", command=lambda:selecionar_filtro_pa(1)) 
submenu_passa_alta.add_separator()
submenu_passa_alta.add_command(label="Filtro Laplaciano", command=lambda:selecionar_filtro_pa(2))

# Adiciona os submenus ao menu principal Filtros
filtros_menu.add_cascade(label="Passa-Baixa", menu=submenu_passa_baixa)
filtros_menu.add_cascade(label="Passa-Alta", menu=submenu_passa_alta)

#Menu de Segmentação
segmentacao_menu = tk.Menu(menu_apk, tearoff=0)
menu_apk.add_cascade(label="Segmentação", menu=segmentacao_menu)
segmentacao_menu.add_command(label="Limiarização Binária", command=lambda:exibir_sliders_binaria())
segmentacao_menu.add_separator()
segmentacao_menu.add_command(label="Limiarização Adaptativa", command=lambda:exibir_sliders_adaptativa())

#Menu da Morfologia
morfologia_menu = tk.Menu(menu_apk, tearoff=0)
menu_apk.add_cascade(label="Operações Morfológias", menu=morfologia_menu)
morfologia_menu.add_command(label="Erodir", command=lambda:exibir_sliders_erodir())
morfologia_menu.add_separator()
morfologia_menu.add_command(label="Dilatar", command=lambda:exibir_sliders_dilatar())

#Cria a tela onde fica a imagem original
tela_imagem_original = tk.Canvas(aplicativo, width=500, height=550, bg="#2e2e2e", highlightthickness=1, highlightbackground="gray")
tela_imagem_original.grid(row=0, column=0, padx=20, pady=20) #padx e pady define o espaçamento horizontal e vertical da tela

#Cria a tela onde fica a imagem editada
tela_imagem_editada = tk.Canvas(aplicativo, width=500, height=550, bg="#2e2e2e", highlightthickness=1, highlightbackground="gray")
tela_imagem_editada.grid(row=0, column=1, padx=20, pady=20) #aqui o column é 1, pois vão ficar duas telas uma do lado da outra (0x0 e 0x1)

# Variável para armazenar o filtro selecionado
filtro_selecionado = 0

#Cria o slider do kernel (passa-baixa)
frame_slider_kernel_pb = tk.Frame(aplicativo, bg="#2e2e2e")

label_kernel_pb = tk.Label(frame_slider_kernel_pb, text="Tamanho do Kernel (passa-baixa):", bg="#2e2e2e", fg="white")
label_kernel_pb.grid(row=0, column=0, padx=5)

slider_kernel_pb = tk.Scale(frame_slider_kernel_pb, from_=1, to=31, resolution=2,orient=tk.HORIZONTAL, length=300, bg="#2e2e2e", fg="white",
                         highlightbackground="#2e2e2e", command=lambda val: atualizar_filtro_passa_baixa(filtro_selecionado))
slider_kernel_pb.set(1)
slider_kernel_pb.grid(row=0, column=1, padx=5)

# Cria o slider do kernel (passa-alta)
frame_slider_kernel_pa = tk.Frame(aplicativo, bg="#2e2e2e")

label_kernel_pa = tk.Label(frame_slider_kernel_pa, text="Tamanho do Kernel (passa-alta):", bg="#2e2e2e", fg="white")
label_kernel_pa.grid(row=0, column=0, padx=5)

slider_kernel_pa = tk.Scale(frame_slider_kernel_pa, from_=1, to=31, resolution=2, orient=tk.HORIZONTAL, length=300, bg="#2e2e2e", fg="white",
                             highlightbackground="#2e2e2e", command=lambda val: atualizar_filtro_passa_alta(filtro_selecionado))
slider_kernel_pa.set(3)
slider_kernel_pa.grid(row=0, column=1, padx=5)

#Cria a interface para o slider do sigma (Filtro de Gauss)
frame_sigma = tk.Frame(aplicativo, bg="#2e2e2e")
label_sigma = tk.Label(frame_sigma, text="Sigma (Desvio Padrão):", bg="#2e2e2e", fg="white")
label_sigma.grid(row=0, column=0, padx=5)
slider_sigma = tk.Scale(frame_sigma, from_=0, to=10, resolution=0.1, orient=tk.HORIZONTAL, length=300, bg="#2e2e2e", fg="white",
                        highlightbackground="#2e2e2e", command=lambda val: atualizar_filtro_passa_baixa(filtro_selecionado))
slider_sigma.set(0)
slider_sigma.grid(row=0, column=1, padx=5)

#Cria a interface para ajustar a direção do sobel
frame_sobel = tk.Frame(aplicativo, bg="#2e2e2e")
label_sobel = tk.Label(frame_sobel, text="Direção Sobel:", bg="#2e2e2e", fg="white")
label_sobel.grid(row=0, column=0, padx=5)
sobel_var = tk.StringVar(value="Ambos")
sobel_menu = tk.OptionMenu(frame_sobel, sobel_var, "Horizontal", "Vertical", "Ambos",
                           command=lambda val: atualizar_filtro_passa_alta(0))
sobel_menu.grid(row=0, column=1, padx=5)

#Cria a interface para o slider do threshold (Filtro de Canny)
frame_canny = tk.Frame(aplicativo, bg="#2e2e2e")
label_canny1 = tk.Label(frame_canny, text="Threshold 1:", bg="#2e2e2e", fg="white")
label_canny1.grid(row=0, column=0, padx=5)
slider_canny1 = tk.Scale(frame_canny, from_=0, to=255, orient=tk.HORIZONTAL, length=300, bg="#2e2e2e", fg="white",
                         highlightbackground="#2e2e2e", command=lambda val: atualizar_filtro_passa_alta(1))
slider_canny1.set(50)
slider_canny1.grid(row=0, column=1, padx=5)
 
label_canny2 = tk.Label(frame_canny, text="Threshold 2:", bg="#2e2e2e", fg="white")
label_canny2.grid(row=1, column=0, padx=5)
slider_canny2 = tk.Scale(frame_canny, from_=0, to=255, orient=tk.HORIZONTAL, length=300, bg="#2e2e2e", fg="white",
                         highlightbackground="#2e2e2e", command=lambda val: atualizar_filtro_passa_alta(1))
slider_canny2.set(100)
slider_canny2.grid(row=1, column=1, padx=5)

#Cria a interface para Limiarização Binária
frame_binaria = tk.Frame(aplicativo, bg="#2e2e2e")
label_thresh = tk.Label(frame_binaria, text="Valor do Limiar:", bg="#2e2e2e", fg="white")
label_thresh.grid(row=0, column=0, padx=5)
slider_thresh = tk.Scale(frame_binaria, from_=0, to=255, orient=tk.HORIZONTAL, length=300, bg="#2e2e2e", fg="white",
                         highlightbackground="#2e2e2e", command=lambda val: aplicar_limiarizacao_binaria(int(val), slider_maxval.get()))
slider_thresh.set(75)
slider_thresh.grid(row=0, column=1, padx=5)

label_maxval = tk.Label(frame_binaria, text="Valor Máximo:", bg="#2e2e2e", fg="white")
label_maxval.grid(row=1, column=0, padx=5)
slider_maxval = tk.Scale(frame_binaria, from_=0, to=255, orient=tk.HORIZONTAL, length=300, bg="#2e2e2e", fg="white",
                         highlightbackground="#2e2e2e", command=lambda val: aplicar_limiarizacao_binaria(slider_thresh.get(), int(val)))
slider_maxval.set(255)
slider_maxval.grid(row=1, column=1, padx=5)

#Cria a interface para Limiarização Adaptativa
frame_adaptativa = tk.Frame(aplicativo, bg="#2e2e2e")
label_blockSize = tk.Label(frame_adaptativa, text="Tamanho do Bloco:", bg="#2e2e2e", fg="white")
label_blockSize.grid(row=0, column=0, padx=5)
slider_blockSize = tk.Scale(frame_adaptativa, from_=3, to=51, resolution=2, orient=tk.HORIZONTAL, length=300, bg="#2e2e2e", fg="white",
                            highlightbackground="#2e2e2e", command=lambda val: aplicar_limiarizacao_adaptativa(int(val), slider_C.get()))
slider_blockSize.set(51)
slider_blockSize.grid(row=0, column=1, padx=5)

label_C = tk.Label(frame_adaptativa, text="Constante (C):", bg="#2e2e2e", fg="white")
label_C.grid(row=1, column=0, padx=5)
slider_C = tk.Scale(frame_adaptativa, from_=0, to=50, orient=tk.HORIZONTAL, length=300, bg="#2e2e2e", fg="white",
                    highlightbackground="#2e2e2e", command=lambda val: aplicar_limiarizacao_adaptativa(slider_blockSize.get(), int(val)))
slider_C.set(5)
slider_C.grid(row=1, column=1, padx=5)

# Criação da interface para Erosão
frame_erodir = tk.Frame(aplicativo, bg="#2e2e2e")
label_erodir_kernel = tk.Label(frame_erodir, text="Tamanho do Kernel (Erosão):", bg="#2e2e2e", fg="white")
label_erodir_kernel.grid(row=0, column=0, padx=5)
slider_erodir_kernel = tk.Scale(frame_erodir, from_=1, to=31, resolution=2, orient=tk.HORIZONTAL, length=300, bg="#2e2e2e", fg="white",
                                highlightbackground="#2e2e2e", command=lambda val: aplicar_erodir(int(val)))
slider_erodir_kernel.set(3)
slider_erodir_kernel.grid(row=0, column=1, padx=5)

# Criação da interface para Dilatação
frame_dilatar = tk.Frame(aplicativo, bg="#2e2e2e")
label_dilatar_kernel = tk.Label(frame_dilatar, text="Tamanho do Kernel (Dilatação):", bg="#2e2e2e", fg="white")
label_dilatar_kernel.grid(row=0, column=0, padx=5)
slider_dilatar_kernel = tk.Scale(frame_dilatar, from_=1, to=31, resolution=2, orient=tk.HORIZONTAL, length=300, bg="#2e2e2e", fg="white",
                                 highlightbackground="#2e2e2e", command=lambda val: aplicar_dilatar(int(val)))
slider_dilatar_kernel.set(3)
slider_dilatar_kernel.grid(row=0, column=1, padx=5)

aplicativo.mainloop()

