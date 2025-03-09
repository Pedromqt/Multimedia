import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
import scipy.fftpack

QY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])
QCbCr = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])
Cb_Down = np.array([[136.037 ,135.862 ,135.869 ,135.875 ,135.875 ,135.869 ,135.869 ,135.875],
    [136.037 ,135.869 ,135.869 ,135.875 ,135.875 ,135.869 ,135.869 ,135.875],
    [136.037 ,135.869, 135.869 ,136.037 ,136.037 ,135.869 ,135.869 ,136.037],
    [134.706 ,134.869 ,134.869 ,134.869 ,134.869 ,134.862 ,134.862 ,135.869],
    [134.869 ,134.869 ,134.869 ,134.862 ,134.862 ,134.862 ,134.862 ,134.862],
    [134.031 ,134.369 ,134.031 ,134.194 ,134.194 ,134.031 ,134.031 ,134.194],
    [133.194 ,133.031 ,133.031 ,132.856 ,134.187 ,134.031 ,134.031 ,134.194],
    [132.856 ,133.031 ,133.031 ,133.019 ,133.019 ,133.031 ,133.031 ,133.194]])

YCbCr_matrix = np.array([[0.299,0.587,0.114],[-0.168736,-0.331264,0.5],[0.5,-0.418688,-0.081312]])
YCbCr_matrix_2 = np.array([0, 128, 128])
cm_red = clr.LinearSegmentedColormap.from_list("red",[(0,0,0),(1,0,0)], N=256)
cm_green = clr.LinearSegmentedColormap.from_list("green",[(0,0,0),(0,1,0)], N=256)
cm_blue = clr.LinearSegmentedColormap.from_list("blue",[(0,0,0),(0,0,1)], N=256)
cm_grey = clr.LinearSegmentedColormap.from_list("grey",[(0,0,0),(1,1,1)], N=256)



#resize img, none, escala h, escala v, (linear, nearest,cubico, area)
def showImg(img,title,cmap=None):
    plt.figure()
    plt.imshow(img,cmap)
    plt.axis("off")
    plt.title(title)
    plt.show()

def showImgLog(img, title, cmap=None):
    img_log = np.log(np.abs(img) + 0.0001)
    plt.imshow(img_log, cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()

def showSubMatrix(img,i,j,dim):
    nd = img.ndim # numero de dimensoes da matriz
    if nd==2:
        img = img.astype(np.float32)
        print(img[i:i+dim,j:j+dim])
    elif nd==3:
        img = img.astype(np.float32)
        print(img[i:i+dim,j:j+dim,0])
        
def downsampling(Y,Cb,Cr):
    Y_d = Y
    showSubMatrix(Cb,8,8,8)
    Cb_dA = cv2.resize(Cb, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    Cr_dA = cv2.resize(Cr, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    erroA = np.mean(np.abs(Cb_dA[8:16,8:16] - Cb_Down))
    print("Matriz canal Cb Interpolacao Area :\n")
    showSubMatrix(Cb_dA,8,8,8)
    Cb_dN = cv2.resize(Cb, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    Cr_dN = cv2.resize(Cr, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    erroN = np.mean(np.abs(Cb_dN[8:16,8:16] - Cb_Down))
    print("Matriz canal Cb Interpolacao Nearest :\n")
    showSubMatrix(Cb_dN,8,8,8)
    Cb_dL = cv2.resize(Cb, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    Cr_dL = cv2.resize(Cr, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    erroL = np.mean(np.abs(Cb_dL[8:16,8:16] - Cb_Down))
    print("Matriz canal Cb Interpolacao Linear :\n")
    showSubMatrix(Cb_dL,8,8,8)
    Cb_dC = cv2.resize(Cb, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    Cr_dC = cv2.resize(Cr, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    erroC = np.mean(np.abs(Cb_dC[8:16,8:16] - Cb_Down))
    print("Matriz canal Cb Interpolacao Cubica :\n")
    showSubMatrix(Cb_dC,8,8,8)
    print("Erro comparardo com a Matriz Teorica :\n")
    print(erroC)
    Cb_dS4 = cv2.resize(Cb, None, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)
    Cr_dS4 = cv2.resize(Cr, None, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)
    erroS4 = np.mean(np.abs(Cb_dS4[8:16,8:16] - Cb_Down))
    print("Matriz canal Cb Interpolacao LancZOS4 :\n")
    showSubMatrix(Cb_dS4,8,8,8)
    print("Erro comparardo com a Matriz Teorica :\n")
    print(erroS4)
    return  Y_d, Cb_dL, Cr_dL

def upsampling(Y,Cb,Cr):
    Cb2  = cv2.resize(Cb, None, fx=1/fx, fy=1/fy, interpolation=cv2.INTER_LINEAR)
    Cr2  = cv2.resize(Cr, None, fx=1/fx, fy=1/fy, interpolation=cv2.INTER_LINEAR)
    imgRec = np.stack((Y,Cb2,Cr2), axis = -1)
    return imgRec

def dpcm_encode(coef_dc):
    diff = coef_dc.copy()
    for i in range(0, len(coef_dc), 8):
        for j in range(0, len(coef_dc[0]), 8):
            if j == 0:
                if i != 0:
                    diff[i][j] = coef_dc[i][j] - coef_dc[i-8][0]
            else:
                diff[i][j] = coef_dc[i][j] - coef_dc[i][j-8]
    return diff
    
def dpcm_decode(diff):
    coef_dc = diff.copy()
    for i in range(0, len(diff), 8):
        for j in range(0, len(diff[0]), 8):
            if j == 0:
                if i != 0:
                    coef_dc[i][j] = coef_dc[i -8][0] + diff[i][j]
            else:
                coef_dc[i][j] = coef_dc[i][j-8] + diff[i][j]
    return coef_dc


def dct_quantize(Y_dct, Cb_dct, Cr_dct, Qualidade):
    global QY, QCbCr
    h, w = Y_dct.shape
    h_c, w_c = Cb_dct.shape

    if Qualidade >= 50:
        FatorEscala = (100 - Qualidade) / 50
    else:
        FatorEscala = 50 / Qualidade
    
  
    QY = np.clip(np.round(QY * FatorEscala), 1, 255).astype(np.uint8)
    QCbCr = np.clip(np.round(QCbCr * FatorEscala), 1, 255).astype(np.uint8)
    
    print("Matriz QY:\n")
    print(QY)
   
    Yb_Q = np.zeros_like(Y_dct, dtype=np.float32)
    for i in range(0, h, 8):  
        for j in range(0, w, 8): 
            Yb_Q[i:i+8, j:j+8] = np.round(Y_dct[i:i+8, j:j+8] / QY).astype(np.float32)

    Cbb_Q = np.zeros_like(Cb_dct, dtype=np.float32)
    Crb_Q = np.zeros_like(Cr_dct, dtype=np.float32)
    for i in range(0, h_c, 8):
        for j in range(0, w_c, 8):
            Cbb_Q[i:i+8, j:j+8] = np.round(Cb_dct[i:i+8, j:j+8] / QCbCr).astype(np.float32)
            Crb_Q[i:i+8, j:j+8] = np.round(Cr_dct[i:i+8, j:j+8] / QCbCr).astype(np.float32)

    
    
    print("Matriz Yb_Q quantizada: \n")
    showSubMatrix(Yb_Q, 8, 8, 8)
    #showImgLog(Yb_Q, "Yb_Q", cm_grey)
    #showImgLog(Cbb_Q, "Cbb_Q", cm_grey)
    #showImgLog(Crb_Q, "Crb_Q", cm_grey)
    
    return Yb_Q, Cbb_Q, Crb_Q



def dct_dequantize(Yb_dct,Cbb_dct,Crb_dct):
    h, w = Yb_dct.shape
    h_c, w_c = Cbb_dct.shape
   
    Yb_Q = np.zeros_like(Yb_dct, dtype=np.float32)
    for i in range(0, h, 8):  
        for j in range(0, w, 8): 
            Yb_Q[i:i+8, j:j+8] = np.round(Yb_dct[i:i+8, j:j+8] * QY).astype(np.float32)

    Cbb_Q = np.zeros_like(Cbb_dct, dtype=np.float32)
    Crb_Q = np.zeros_like(Crb_dct, dtype=np.float32)
    for i in range(0, h_c, 8):
        for j in range(0, w_c, 8):
            Cbb_Q[i:i+8, j:j+8] = np.round(Cbb_dct[i:i+8, j:j+8] * QCbCr).astype(np.float32)
            Crb_Q[i:i+8, j:j+8] = np.round(Crb_dct[i:i+8, j:j+8] * QCbCr).astype(np.float32)
        
    return Yb_Q, Cbb_Q, Crb_Q


def dct_calc_blocks(channel, number_blocks):
    
    h, w = channel.shape
    channel_dct = np.zeros_like(channel, dtype=np.float32)

    
    if number_blocks == 64 and w % 64 != 0:
        last_column = channel[:, -1:]
        channel = np.hstack((channel, np.tile(last_column, (1, 32))))

    
    for i in range(0, h, number_blocks):
        for j in range(0, w, number_blocks):
            block = channel[i:i+number_blocks, j:j+number_blocks]
            block_dct = scipy.fftpack.dct(scipy.fftpack.dct(block, norm="ortho").T, norm="ortho").T
            channel_dct[i:i+number_blocks, j:j+number_blocks] = block_dct

    return channel_dct


def dct_inv_blocks(channel_dct, number_blocks):
    h, w = channel_dct.shape
    channel_idct = np.zeros_like(channel_dct, dtype=np.float32)

    
    for i in range(0, h, number_blocks):
        for j in range(0, w, number_blocks):
            block = channel_dct[i:i+number_blocks, j:j+number_blocks]
            block_idct = scipy.fftpack.idct(scipy.fftpack.idct(block.T, norm="ortho").T, norm="ortho")
            channel_idct[i:i+number_blocks, j:j+number_blocks] = block_idct

    
    if number_blocks == 64 and w != 1216:
        channel_idct = channel_idct[:, :-32]

    return channel_idct


def dct_calc8(Y_d,Cb_d,Cr_d):
    Y_dct8 = dct_calc_blocks(Y_d,8)
    Cb_dct8 = dct_calc_blocks(Cb_d,8)
    Cr_dct8 = dct_calc_blocks(Cr_d,8)
    return Y_dct8, Cb_dct8, Cr_dct8

def dct_inv8(Y_dct8,Cb_dct8,Cr_dct8):
    Y_d = dct_inv_blocks(Y_dct8,8)
    Cb_d = dct_inv_blocks(Cb_dct8,8)
    Cr_d = dct_inv_blocks(Cr_dct8,8)
    return Y_d, Cb_d, Cr_d

def dct_calc(Y_d, Cb_d, Cr_d):
    Y_dct = scipy.fftpack.dct(scipy.fftpack.dct(Y_d, norm="ortho").T, norm="ortho").T
    Cb_dct = scipy.fftpack.dct(scipy.fftpack.dct(Cb_d, norm="ortho").T, norm="ortho").T
    Cr_dct = scipy.fftpack.dct(scipy.fftpack.dct(Cr_d, norm="ortho").T, norm="ortho").T
    
    showSubMatrix(Cb_d, 8, 8, 8)
    #showImgLog(Y_dct, "Y_DCT", cm_grey)
    #showImgLog(Cb_dct, "Cb_DCT", cm_grey)
    #showImgLog(Cr_dct, "Cr_DCT", cm_grey)
    return Y_dct, Cb_dct, Cr_dct


def dct_inv(Y_dct, Cb_dct, Cr_dct):
    Y_d = scipy.fftpack.idct(scipy.fftpack.idct(Y_dct.T, norm="ortho").T, norm="ortho")
    Cb_d = scipy.fftpack.idct(scipy.fftpack.idct(Cb_dct.T, norm="ortho").T, norm="ortho")
    Cr_d = scipy.fftpack.idct(scipy.fftpack.idct(Cr_dct.T, norm="ortho").T, norm="ortho")
    return Y_d, Cb_d, Cr_d

def add_padding(img):
    # adicionar linhas ou colunas = quociente -  resto -> np.repeat -> np.vstack -> np.hstack
    resto_nl = nl % 32
    resto_nc = nc % 32
    if(resto_nl != 0):
        add_nl = 32 - resto_nl
        last_line = img[-1:,:,:]
        array_add_nl = np.repeat(last_line,add_nl,axis=0)
        img = np.vstack((img,array_add_nl))
    if (resto_nc != 0):
        add_nc = 32 - resto_nc
        last_column = img[:,-1:,:]
        array_add_nc = np.repeat(last_column,add_nc,axis=1)
        img = np.hstack((img,array_add_nc))
    #print(img.shape)
    return img
    
def remove_padding(imgRec):
    imgRec = imgRec[0:nl,0:nc,:]
    return imgRec

def YCbCr(img):
    img = img.astype(np.float32)
    img = np.dot(img, YCbCr_matrix.T) + YCbCr_matrix_2
    Y = img[:,:,0]
    Cb = img[:,:,1]
    Cr = img[:,:,2]
    return Y,Cb,Cr

# [Y Cb Cr] -> ao aplicar Transposta: [ Y  ]
#                                     | Cb |
#                                     [ Cr ]
    
def remove_YCbCr(img):
    img -= YCbCr_matrix_2
    remove_YCbCr_matrix = np.linalg.inv(YCbCr_matrix)
    img = np.dot(img, remove_YCbCr_matrix.T)
    img = np.round(img)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def metricas(img,imgRec):
    img=img.astype(np.float32)
    imgRec=imgRec.astype(np.float32)
    mse = np.sum((img-imgRec)**2)/ (img.shape[0]* img.shape[1])
    rmse = np.sqrt(mse)
    p = np.sum(img **2)/(img.shape[0]*img.shape[1])
    snr=10*np.log10(p/mse)
    psnr=10*np.log10((np.max(img)**2)/mse)
    Y,Cb,Cr = YCbCr(img)
    Y_r,Cb_r,Cr_r = YCbCr(imgRec)
    dif_max = np.max(np.abs(Y-Y_r))
    dif_mean = np.mean(np.abs(Y-Y_r))
    print("MSE: ",mse)
    print("RMSE ",rmse)
    print("SNR: ",snr)
    print("PSNR: ",psnr)
    print("MAX Y: ",dif_max)
    print("AVG Y: ",dif_mean)
    return mse, rmse,snr,psnr, dif_max,dif_mean

def encoder(img):
    img = add_padding(img)
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    ##showImg(img,"Imagem com padding")
    ##showImg(R,"Red",cm_red)
    ##showImg(G,"Green",cm_green)
    ##showImg(B,"Blue",cm_blue) 
    #print("Matriz R")  
    #showSubMatrix(R,8,8,8)
    Y,Cb,Cr = YCbCr(img)

    ##showImg(Y,"Y",cm_grey)
    ##showImg(Cb,"Cb",cm_grey)
    ##showImg(Cr,"Cr",cm_grey)
    global fx
    global fy
    fx = 0.5
    fy = 0.5
    Y2,Cb2,Cr2 = downsampling(Y,Cb,Cr)
    print("Tamanho canal Y com downsampling 4:2:0")
    print(Y2.shape[0], Y2.shape[1])
    print("Matriz canal Y :\n")
    print(Y2)
    print("Tamanho canal Cb com downsampling 4:2:0")
    print(Cb2.shape[0], Cb2.shape[1])
    print("Matriz canal Cb :\n")
    print(Cb2)
    print("Tamanho canal Cr com downsampling 4:2:0")
    print(Cr2.shape[0], Cr2.shape[1])
    print("Matriz canal Cr :\n")
    print(Cr2)
   # showImg(Y2,"Y downsampling 4:2:0",cm_grey)
   # showImg(Cb2,"Cb downsampling 4:2:0",cm_grey)
   # showImg(Cr2,"Cr downsampling 4:2:0",cm_grey)
    fx = 0.5
    fy = 1
    Y_d,Cb_d,Cr_d = downsampling(Y,Cb,Cr)
    
   # showImg(Y_d,"Y downsampling 4:2:2",cm_grey)
   # showImg(Cb_d,"Cb downsampling 4:2:2",cm_grey)
   # showImg(Cr_d,"Cr downsampling 4:2:2",cm_grey)
    print("Tamanho canal Y com downsampling 4:2:2")
    print(Y.shape[0], Y.shape[1])
    print("Matriz canal Y :\n")
    showSubMatrix(Y_d, 8, 8, 8)
    print("Tamanho canal Cb com downsampling 4:2:2")
    print(Cb_d.shape[0], Cb_d.shape[1])
    print("Matriz canal Cb :\n")
    showSubMatrix(Cb_d, 8, 8, 8)
    print("Tamanho canal Cr com downsampling 4:2:2")
    print(Cr_d.shape[0], Cr_d.shape[1])
    print("Matriz canal Cr :\n")
    showSubMatrix(Cr_d, 8, 8, 8)
    
    Y_dct, Cb_dct, Cr_dct = dct_calc(Y_d,Cb_d,Cr_d)
   
    print("Matriz Y antes de DCT:\n")
    showSubMatrix(Y_d, 8, 8, 8)
    print("Matriz Y depois de DCT:\n")
    showSubMatrix(Y_dct, 8, 8, 8)
    
    print("Matriz Cb antes de DCT:\n")
    showSubMatrix(Cb_d, 8, 8, 8)
    print("Matriz Cb depois de DCT:\n")
    showSubMatrix(Cb_dct, 8, 8, 8)
    

    print("Matriz Cr antes de DCT:\n")
    showSubMatrix(Cr_d, 8, 8, 8)
    print("Matriz Cr depois de DCT:\n")
    showSubMatrix(Cr_dct, 8, 8, 8)
    
    
    Y_dct8, Cb_dct8, Cr_dct8 = dct_calc8(Y_d,Cb_d,Cr_d)
    
    
    print("Matriz Y antes de DCT8:\n" )
    showSubMatrix(Y_d, 8, 8, 8)
    print("Matriz Y depois de DCT8:\n" )
    showSubMatrix(Y_dct8, 8, 8, 8)
    
    
    print("Matriz Cb antes de DCT:\n" )
    showSubMatrix(Cb_d, 8, 8, 8)
    print("Matriz Cb depois de DCT8:\n" )
    showSubMatrix(Cb_dct8, 8, 8, 8)
    

    print("Matriz Cr antes de DCT8:\n" )
    showSubMatrix(Cr_d, 8, 8, 8)
    print("Matriz Cr depois de DCT8:\n" )
    showSubMatrix(Cr_dct8, 8, 8, 8)
    

    #showImgLog(Y_dct8, "Yb_DCT", cm_grey)
    #showImgLog(Cb_dct8, "Cbb_DCT", cm_grey)
    #showImgLog(Cr_dct8, "Crb_DCT", cm_grey)
    
    
    Yb_Q, Cbb_Q, Crb_Q = dct_quantize(Y_dct8, Cb_dct8, Cr_dct8, qualidade)

    
    Yb_DPCM = dpcm_encode(Yb_Q)
    Cbb_DPCM = dpcm_encode(Cbb_Q)
    Crb_DPCM = dpcm_encode(Crb_Q)
    
    print("Matriz Y antes de DPCM:\n" )
    showSubMatrix(Yb_Q, 8, 8, 8)
    print("Matriz Y depois de DPCM:\n" )
    showSubMatrix(Yb_DPCM, 8, 8, 8)
    
    
    print("Matriz Cb antes de DPCM:\n")
    showSubMatrix(Cbb_Q, 8, 8, 8)
    print("Matriz Cb depois de DPCM:\n" )
    showSubMatrix(Cbb_DPCM, 8, 8, 8)
   

    print("Matriz Cr antes de DPCM:\n")
    showSubMatrix(Crb_Q, 8, 8, 8)
    print("Matriz Cr depois de DPCM:\n")
    showSubMatrix(Crb_DPCM, 8, 8, 8)
    
    
   # showImgLog(Yb_DPCM, "Yb_DCPM", cm_grey)
    #showImgLog(Cbb_DPCM, "Cbb_DCPM", cm_grey)
    #showImgLog(Crb_DPCM, "Crb_DCPM", cm_grey)
    
    #Y,Cb,Cr = downsampling(Y,Cb,Cr, 0.5, 0.5)
    ##showImg(Y,"Y downsampling 4:2:0",cm_grey)
    ##showImg(Cb,"Cb downsampling 4:2:0",cm_grey)
    ##showImg(Cr,"Cr downsampling 4:2:0",cm_grey)
    #print("------------")
    #print("Matriz Y")
    #showSubMatrix(Y,8,8,8)
    #print("------------")
    #print("Matriz Cb")
    #showSubMatrix(Cb,8,8,8)
    return Yb_DPCM, Cbb_DPCM, Crb_DPCM

def decoder(Y, Cb, Cr):
    Y_Q = dpcm_decode(Y)
    print("Matriz Y depois de DPCM inversa:\n")
    showSubMatrix(Y_Q, 8, 8, 8)
    Cb_Q = dpcm_decode(Cb)
    print("Matriz Cb depois de DPCM inversa:\n")
    showSubMatrix(Cb_Q, 8, 8, 8)
    Cr_Q = dpcm_decode(Cr)
    print("Matriz Cr depois de DPCM inversa:\n")
    showSubMatrix(Cr_Q, 8, 8, 8)
    Y_d, Cb_d, Cr_d = dct_dequantize(Y_Q,Cb_Q,Cr_Q)
    print("Matriz Y depois de dequantize:\n")
    showSubMatrix(Y_d, 8, 8, 8)
    Y, Cb, Cr = dct_inv8(Y_d, Cb_d, Cr_d)
    print("Matriz Y depois de DCT8 inversa:\n")
    showSubMatrix(Y, 8, 8, 8)
    print("Matriz Cb depois de DCT8 inversa:\n")
    showSubMatrix(Cb, 8, 8, 8)
    print("Matriz Cr depois de DCT8 inversa:\n")
    showSubMatrix(Cr, 8, 8, 8)
    imgRec = upsampling(Y, Cb, Cr)
    print("Matriz canal Cb upsampling:\n")
    showSubMatrix(imgRec[:,:,1], 8, 8, 8)
    imgRec = remove_padding(imgRec)
    imgRec = remove_YCbCr(imgRec)
    return imgRec

def main():
    fName = "./imagens/airport.bmp"
    fName1 = "./imagens/geometric.bmp"
    fName2 = "./imagens/nature.bmp"
    img = plt.imread(fName)
    img1 = plt.imread(fName1)
    img2 = plt.imread(fName2)
    global nl,nc, qualidade
    qualidadeArray = [10,25,50,75,100]
    showImg(img,"Imagem Original")

    for qualidade in qualidadeArray:
        img_copy = img.copy()
        nl,nc,_= img.shape
        Y, Cb, Cr = encoder(img_copy)
        imgRec = decoder(Y, Cb, Cr)
        showImg(imgRec, f"Imagem Reconstruida com qualidade {qualidade}")
        metricas(img, imgRec)

    showImg(img1,"Imagem Original")

    for qualidade in qualidadeArray:
        img_copy = img1.copy()
        nl,nc,_= img1.shape  
        Y, Cb, Cr = encoder(img_copy)
        imgRec = decoder(Y, Cb, Cr)
        showImg(imgRec, f"Imagem Reconstruida com qualidade {qualidade}")
        metricas(img1, imgRec)

    showImg(img2,"Imagem Original")

    for qualidade in qualidadeArray:
        img_copy = img2.copy()
        nl,nc,_= img2.shape  
        Y, Cb, Cr = encoder(img_copy)
        imgRec = decoder(Y, Cb, Cr)
        showImg(imgRec, f"Imagem Reconstruida com qualidade {qualidade}")
        metricas(img2, imgRec)

    print("R_decoded:\n")
    showSubMatrix(imgRec[:,:,0], 8, 8, 8)
    

if __name__ == "__main__":
    main()