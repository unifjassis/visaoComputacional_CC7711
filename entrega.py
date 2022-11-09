import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


def aviao():
    # Importa e converta para RGB
    img = cv2.imread('./imagens/AVIAO_01.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convertendo para preto e branco (RGB -> Gray Scale -> BW)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    a = img_gray.max()
    _, thresh = cv2.threshold(img_gray, a / 2 * 1.7, a, cv2.THRESH_BINARY_INV)

    tamanhoKernel = 5
    kernel = np.ones((tamanhoKernel, tamanhoKernel), np.uint8)
    thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Filtro de ruído (bluring)
    img_blur = cv2.blur(img_gray, ksize=(tamanhoKernel, tamanhoKernel))

    # Detecção borda com Canny (sem blurry)
    edges_gray = cv2.Canny(image=img_gray, threshold1=a / 2, threshold2=a / 2)
    # Detecção borda com Canny (com blurry)
    edges_blur = cv2.Canny(image=img_blur, threshold1=a / 2, threshold2=a / 2)

    # contorno
    contours, hierarchy = cv2.findContours(
        image=edges_gray,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_copy = img.copy()
    final = cv2.drawContours(img_copy, contours, contourIdx=-1,
                             color=(255, 0, 0), thickness=1)

    # plot imagens
    imagens2 = [img, img_blur, img_gray, edges_gray, edges_blur, thresh, thresh_open]
    imagens = final

    formatoX = math.ceil(len(imagens2) ** .5)

    if (formatoX ** 2 - len(imagens2)) > formatoX:
        formatoY = formatoX - 1
    else:
        formatoY = formatoX
    for i in range(len(imagens2)):
        plt.subplot(formatoY, formatoX, i + 1)
        plt.imshow(imagens2[i], 'gray')
        plt.xticks([]), plt.yticks([])

    plt.show()

    plt.imshow(imagens)
    plt.show()


def castelo():
    # Importa e converta para RGB
    img = cv2.imread('./imagens/CASTELO_01.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convertendo para preto e branco (RGB -> Gray Scale -> BW)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    a = img_gray.max()
    _, thresh = cv2.threshold(img_gray, a / 2 * 1.7, a, cv2.THRESH_BINARY_INV)

    tamanhoKernel = 5
    kernel = np.ones((tamanhoKernel, tamanhoKernel), np.uint8)
    thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Filtro de ruído (bluring)
    img_blur = cv2.blur(img_gray, ksize=(tamanhoKernel, tamanhoKernel))

    # Detecção borda com Canny (sem blurry)
    edges_gray = cv2.Canny(image=img_gray, threshold1=a / 2, threshold2=a / 2)
    # Detecção borda com Canny (com blurry)
    edges_blur = cv2.Canny(image=img_blur, threshold1=a / 2, threshold2=a / 2)

    # contorno
    contours, hierarchy = cv2.findContours(
        image=edges_gray,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_copy = img.copy()
    final = cv2.drawContours(img_copy, contours, contourIdx=-1,
                             color=(255, 0, 0), thickness=1)

    # plot imagens
    imagens2 = [img, img_blur, img_gray, edges_gray, edges_blur, thresh, thresh_open]
    imagens = final

    formatoX = math.ceil(len(imagens2) ** .5)

    if (formatoX ** 2 - len(imagens2)) > formatoX:
        formatoY = formatoX - 1
    else:
        formatoY = formatoX
    for i in range(len(imagens2)):
        plt.subplot(formatoY, formatoX, i + 1)
        plt.imshow(imagens2[i], 'gray')
        plt.xticks([]), plt.yticks([])

    plt.show()

    plt.imshow(imagens)
    plt.show()


aviao()
castelo()
