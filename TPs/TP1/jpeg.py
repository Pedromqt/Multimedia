import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
import scipy
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
        
def downsampling(Y,Cb,Cr, fx, fy):
    Y_d = Y
    Cb_d = cv2.resize(Cb, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    Cr_d = cv2.resize(Cr, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    return  Y_d, Cb_d, Cr_d

def upsampling(Y,Cb,Cr):
    Cb2  = cv2.resize(Cb, None, fx=1/fx, fy=1/fy, interpolation=cv2.INTER_AREA)
    Cr2  = cv2.resize(Cr, None, fx=1/fx, fy=1/fy, interpolation=cv2.INTER_AREA)
    imgRec = np.stack((Y,Cb2,Cr2), axis = -1)
    return imgRec

import numpy as np

def dct_quantize(Y_dct, Cb_dct, Cr_dct, Qualidade):
    global QY,QCbCr
    h, w = Y_dct.shape
    h_c, w_c = Cb_dct.shape
    if(Qualidade >= 50):
        FatorEscala = (100-Qualidade)/50
    else:
        FatorEscala = 50/Qualidade
    if(FatorEscala == 0):
        QY = np.ones((8,8))
        QCbCr = np.ones((8,8))
    else:
        QY = np.round(QY*FatorEscala).astype(np.int32)
        QCbCr = np.round(QCbCr*FatorEscala).astype(np.int32)

    Y_dct_reshaped = Y_dct.reshape(h // 8, 8, w // 8, 8)

    Cb_dct_reshaped = Cb_dct.reshape(h_c // 8, 8, w_c // 8, 8)
    Cr_dct_reshaped = Cr_dct.reshape(h_c // 8, 8, w_c // 8, 8)

    Yb_Q = np.round(Y_dct_reshaped / QY[np.newaxis, :, np.newaxis, :]).astype(np.int32)
    Cbb_Q = np.round(Cb_dct_reshaped / QCbCr[np.newaxis, :,np.newaxis, :]).astype(np.int32)
    Crb_Q = np.round(Cr_dct_reshaped / QCbCr[np.newaxis, :,np.newaxis, :]).astype(np.int32)
    
    Yb_Q = Yb_Q.reshape(h, w)
    Cbb_Q = Cbb_Q.reshape(h_c, w_c)
    Crb_Q = Crb_Q.reshape(h_c, w_c)
    
    showSubMatrix(Yb_Q, 8, 8, 8)
    showSubMatrix(QY, 8, 8, 8)
    
    showImgLog(Yb_Q, "Yb_Q", cm_grey)
    showImgLog(Cbb_Q, "Cbb_Q", cm_grey)
    showImgLog(Crb_Q, "Crb_Q", cm_grey)
    
    return Yb_Q, Cbb_Q, Crb_Q


def dct_dequantize(Yb_dct,Cbb_dct,Crb_dct):
    h, w = Yb_dct.shape
    h_c, w_c = Cbb_dct.shape
    
    Yb_dct_reshaped = Yb_dct.reshape(h // 8, 8, w // 8, 8)
    Cbb_dct_reshaped = Cbb_dct.reshape(h_c // 8, 8, w_c // 8, 8)
    Crb_dct_reshaped = Crb_dct.reshape(h_c // 8, 8, w_c // 8, 8)

    Y_dct = np.round(Yb_dct_reshaped * QY[np.newaxis, :, np.newaxis, :]).astype(np.int32)
    Cb_dct = np.round(Cbb_dct_reshaped * QCbCr[np.newaxis, :,np.newaxis, :]).astype(np.int32)
    Cr_dct = np.round(Crb_dct_reshaped * QCbCr[np.newaxis, :,np.newaxis, :]).astype(np.int32)
    
    Y_dct = Y_dct.reshape(h, w)
    Cb_dct = Cb_dct.reshape(h_c, w_c)
    Cr_dct = Cr_dct.reshape(h_c, w_c)
    return Y_dct, Cb_dct, Cr_dct

def dct_inv_blocks(channel_dct,number_blocks):
    h, w = channel_dct.shape
    channel_dct_blocks = channel_dct.reshape(h // number_blocks, number_blocks, w // number_blocks, number_blocks).transpose(0, 2, 1, 3)
    channel_idct = scipy.fftpack.idct(scipy.fftpack.idct(channel_dct_blocks, axis=2, norm="ortho"), axis=3, norm="ortho")
    return channel_idct.transpose(0, 2, 1, 3).reshape(h, w)

def dct_calc_blocks(channel,number_blocks):
    h, w = channel.shape
    channel_blocks = channel.reshape(h // number_blocks, number_blocks, w // number_blocks, number_blocks).transpose(0, 2, 1, 3)
    channel_dct = scipy.fftpack.dct(scipy.fftpack.dct(channel_blocks, axis=2, norm="ortho"), axis=3, norm="ortho")
    return channel_dct.transpose(0, 2, 1, 3).reshape(h, w) 

def dct_calc8(Y_d,Cb_d,Cr_d): 
    Y_dct8 = dct_calc_blocks(Y_d,8)
    Cb_dct8 = dct_calc_blocks(Cb_d,8)
    Cr_dct8 = dct_calc_blocks(Cr_d,8)
    showImgLog(Y_dct8, "Yb_DCT", cm_grey)
    showImgLog(Cb_dct8, "Cbb_DCT", cm_grey)
    showImgLog(Cr_dct8, "Crb_DCT", cm_grey)
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
    showImgLog(Y_dct, "Y_DCT", cm_grey)
    showImgLog(Cb_dct, "Cb_DCT", cm_grey)
    showImgLog(Cr_dct, "Cr_DCT", cm_grey)
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

def encoder(img):
    img = add_padding(img)
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    #showImg(img,"Imagem com padding")
    #showImg(R,"Red",cm_red)
    #showImg(G,"Green",cm_green)
    #showImg(B,"Blue",cm_blue) 
    #print("Matriz R")  
    #showSubMatrix(R,8,8,8)
    Y,Cb,Cr = YCbCr(img)
    #showImg(Y,"Y",cm_grey)
    #showImg(Cb,"Cb",cm_grey)
    #showImg(Cr,"Cr",cm_grey)
    global fx
    global fy
    fx = 0.5
    fy = 0.5
    Y2,Cb2,Cr2 = downsampling(Y,Cb,Cr, fx, fy)
    showImg(Y2,"Y downsampling 4:2:0",cm_grey)
    showImg(Cb2,"Cb downsampling 4:2:0",cm_grey)
    showImg(Cr2,"Cr downsampling 4:2:0",cm_grey)
    fx = 0.5
    fy = 1
    Y_d,Cb_d,Cr_d = downsampling(Y,Cb,Cr, fx, fy)
    showImg(Y_d,"Y downsampling 4:2:2",cm_grey)
    showImg(Cb_d,"Cb downsampling 4:2:2",cm_grey)
    showImg(Cr_d,"Cr downsampling 4:2:2",cm_grey)

    Y_dct, Cb_dct, Cr_dct = dct_calc(Y_d,Cb_d,Cr_d)
    Y_dct8, Cb_dct8, Cr_dct8 = dct_calc8(Y_d,Cb_d,Cr_d)
    Yb_Q, Cbb_Q, Crb_Q = dct_quantize(Y_dct8, Cb_dct8, Cr_dct8, 75)
    
    
    #Y,Cb,Cr = downsampling(Y,Cb,Cr, 0.5, 0.5)
    #showImg(Y,"Y downsampling 4:2:0",cm_grey)
    #showImg(Cb,"Cb downsampling 4:2:0",cm_grey)
    #showImg(Cr,"Cr downsampling 4:2:0",cm_grey)
    #print("------------")
    #print("Matriz Y")
    #showSubMatrix(Y,8,8,8)
    #print("------------")
    #print("Matriz Cb")
    #showSubMatrix(Cb,8,8,8)
    return Yb_Q, Cbb_Q, Crb_Q

def decoder(Y, Cb, Cr):
    Y, Cb, Cr = dct_dequantize(Y, Cb, Cr)
    Y_d, Cb_d, Cr_d = dct_inv8(Y, Cb, Cr)
    imgRec = upsampling(Y_d, Cb_d, Cr_d)
    imgRec = remove_padding(imgRec)
    imgRec = remove_YCbCr(imgRec)
    return imgRec

def main():
    fName = "./imagens/airport.bmp"
    img = plt.imread(fName)
    global nl,nc
    nl,nc,_= img.shape
    showImg(img,fName)
    Y,Cb,Cr = encoder(img)
    imgRec = decoder(Y,Cb,Cr)
    showImg(imgRec,"Imagem Reconstruida")

if __name__ == "__main__":
    main()