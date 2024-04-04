import numpy as np
import cv2
import sys
import timeit

INPUT_IMG = 'teste.bmp'
JANELA = 5

'''def normalizacaomedia(img):
    min = 1
    max = 1
    range = 0

    altura, largura, canais = img.shape;
    
    max = np.max(img)
    min = np.min(img)

    range = max - min
    
    for i in range(altura):
        for j in range(largura):
            img[i][j] = ((img[i][j] - min) / range)

    cv2.imshow('Import',img);
    cv2.waitKey(0); '''

def algoritmoIngenuo(img, JANELA):

    altura, largura, canais = img.shape;
    

    # Matriz para armazenar as médias das janelas
    img_out = np.zeros((altura, largura, canais), dtype=np.float32)

    start_time = timeit.default_timer ()

    for c in range(canais):
        for i in range(altura):
            for j in range(largura):
            
                soma = 0
                total_pixels = 0
                
                for y in range(max(0, i - JANELA), min(altura, i + JANELA)): #Nesse caso vamos fazer uma média com o que for possível pegar de pixels da borda,
                                                                                # min entre altura e i+janela, vai ficar limitado ao valor maximo da altura e não acessar fora da imagem
                    for x in range(max(0, j - JANELA), min(largura, j + JANELA)): #o maximo que encontrar entre i-Janela e 0, isso sempre vai pegar 0 caso a iteração fosse pegar fora da imagem

                            soma += img[y][x][c]
                            total_pixels += 1

                img_out[i][j][c] = (soma / total_pixels)

    print ('Tempo do Ingenuo: %f' % (timeit.default_timer () - start_time))
    
    cv2.imshow('Blur CV2', IMG_CV)
    cv2.imshow('Blur Ingenuo',img_out)
    cv2.waitKey(0)
 
    

    '''#Colocando as bordas da imagem original

        for i in range(0,JANELA):
            for j in range(0, largura):
                img_out[i][j] = img[i][j];
                img_out[j][i] = img[j][i];
        
        for i in range(altura-JANELA,altura):
            for j in range(0, largura):
                img_out[i][j] = img[i][j];
                img_out[j][i] = img[j][i]; '''

def algoritmo_separavel(img,JANELA):
     
    altura, largura, canais = img.shape
    
    # Matriz para armazenar as médias das janelas
    img_out = np.zeros((altura, largura, canais), dtype=np.float32)

    #Matriz do buffer do filtro
    img_buf = np.zeros((altura, largura, canais), dtype=np.float32)


    #Nessa iteração estamos borrando na horizontal, utilizei a mesma estrutura de acesso das janelas do algoritmo ingenuo

    start_time = timeit.default_timer ()

    for c in range(canais):
        for i in range(altura):
            for j in range(largura):
                soma = 0;
                total_pixels = 0
            
                for ite in range(max(0, j - JANELA), min(largura, j + JANELA)):
                    soma += img[i][ite][c]
                    total_pixels += 1
                
                img_buf[i][j][c] = (soma / total_pixels)
    
    cv2.imshow('Blur Horizontal',img_buf)
    cv2.waitKey(0)
    
    altura1,largura1,canais1 = img_buf.shape
    
    for c in range(canais):
        for i in range(altura1):
            for j in range(largura1):
                soma = 0
                total_pixels = 0

                for ite in range(max(0,i-JANELA), min(altura, i+JANELA)):
                        soma += img_buf[ite][j][c]
                        total_pixels += 1
                
                img_out[i][j][c] = (soma/total_pixels);
    
    print ('Tempo do Separável: %f' % (timeit.default_timer () - start_time))

    cv2.imshow('Blur CV2', IMG_CV)
    cv2.imshow('Blur Filtro Separavel',img_out)
    cv2.waitKey(0)

def img_Integral(img, JANELA):

    altura, largura, canais = img.shape
    #buffer
    img_buf = np.zeros((altura, largura,canais), dtype=np.float32);

    img_blur = np.zeros((altura, largura,canais), dtype=np.float32);

    '''for y in range(1,altura+1):
        for x in range(1, largura+1):
            img_buf[y][x] = (img[y-1][x-1][0])+img_buf[y-1][x]+img_buf[y][x-1] - img_buf[y-1][x-1]'''

    start_time = timeit.default_timer ()

    for c in range(canais):  #For para fazer as operaçoes nos canais de cores
        for i in range(largura):
            img_buf[0][i][c] = img[0][i][c]
            for j in range(1,altura):
                    img_buf[j][i][c] = img [j][i][c] + img_buf[j-1][i][c]
    for c in range(canais): 
        for i in range(1,largura):
            for j in range(altura):
                img_buf[j][i][c]= img_buf[j][i][c] + img_buf[j][i-1][c]

    cv2.imshow('img integral',img_buf/1000);
    cv2.waitKey(0);

    
    for c in range(canais): 
        for i in range(altura):
            for j in range(largura):

                x1 = max(0,i - JANELA) 
                y1 = max(0,j - JANELA)
                x2 = min(altura-1, i + JANELA) 
                y2 = min(largura-1, j + JANELA)

                pixelIntegral = img_buf[x2][y2][c] - img_buf[x2][y1-1][c] - img_buf[x1-1][y2][c] + img_buf[x1-1][y1-1][c]        
                    
                img_blur[i][j][c] = (pixelIntegral / ((x2 - x1+1)*(y2 - y1+1)))

                
    img_blur = img_blur[JANELA+1:, JANELA+1:]

    print (np.min (img_blur))
    print (np.max (img_blur))

    print ('Tempo do Integral: %f' % (timeit.default_timer () - start_time))

    cv2.imshow('img integral',img_blur);
    cv2.imshow('Blur CV2', IMG_CV)
    cv2.waitKey(0)
    
                
img = cv2.imread (INPUT_IMG)
if img is None:
    print("Erro na abertura da imagem")
    sys.exit()

img = img.reshape ((img.shape [0], img.shape [1], 3))
img = img.astype(np.float32)/255

IMG_CV = cv2.blur(img,(((JANELA*2)+1), ((JANELA*2)+1)))# IMG de comparaçao



#fUNÇÕES DOS ALGORITMOS
img_Integral(img, JANELA)
algoritmo_separavel(img,JANELA)
algoritmoIngenuo(img, JANELA)





