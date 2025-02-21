import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
import scipy
import scipy.fftpack

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


def dct_calc(Y, Cb, Cr):
    Y_dct = scipy.fftpack.dct(scipy.fftpack.dct(Y, norm="ortho").T, norm="ortho").T
    Cb_dct = scipy.fftpack.dct(scipy.fftpack.dct(Cb, norm="ortho").T, norm="ortho").T
    Cr_dct = scipy.fftpack.dct(scipy.fftpack.dct(Cr, norm="ortho").T, norm="ortho").T

    showSubMatrix(Cb, 8, 8, 8)
    showImgLog(Y_dct, "Y_DCT", cm_grey)
    showImgLog(Cb_dct, "Cb_DCT", cm_grey)
    showImgLog(Cr_dct, "Cr_DCT", cm_grey)

    return Y_dct, Cb_dct, Cr_dct


def dct_inv(Y_dct, Cb_dct, Cr_dct):
    Y = scipy.fftpack.idct(scipy.fftpack.idct(Y_dct.T, norm="ortho").T, norm="ortho")
    Cb = scipy.fftpack.idct(scipy.fftpack.idct(Cb_dct.T, norm="ortho").T, norm="ortho")
    Cr = scipy.fftpack.idct(scipy.fftpack.idct(Cr_dct.T, norm="ortho").T, norm="ortho")

    return Y, Cb, Cr



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
    Y,Cb,Cr = downsampling(Y,Cb,Cr, fx, fy)
    showImg(Y,"Y downsampling 4:2:2",cm_grey)
    showImg(Cb,"Cb downsampling 4:2:2",cm_grey)
    showImg(Cr,"Cr downsampling 4:2:2",cm_grey)

    dct_calc(Y,Cb,Cr)
    
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
    return Y,Cb,Cr

def decoder(Y,Cb,Cr):
    imgRec = dct_inv(Y,Cb,Cr)
    imgRec = upsampling(Y,Cb,Cr)
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
    showImg(imgRec,"Imagem Reconstruida / sem padding")

if __name__ == "__main__":
    main()