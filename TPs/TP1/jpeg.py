import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
#resize img, none, escala h, escala v, (linear, nearest,cubico, area)
def showImg(img,title,cmap=None):
    plt.figure()
    plt.imshow(img,cmap)
    plt.axis("off")
    plt.title(title)
    plt.show()

def showSubMatrix(img,i,j,dim):
    nd = img.ndim # numero de dimensoes da matriz
    if nd==2:
        img = img.astype(np.float32)
        print(img[i:i+dim,j:j+dim])
    elif nd==3:
        img = img.astype(np.float32)
        print(img[i:i+dim,j:j+dim,0])
        
def downsampling(Cb,Cr, fx, fy):
    Cb = cv2.resize(Cb, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    Cr = cv2.resize(Cr, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    return Cb, Cr

def add_padding(img):
    # adicionar linhas ou colunas = quociente -  resto -> np.repeat -> np.vstack -> np.hstack
    nl,nc,_ = img.shape
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
    return img,add_nl,add_nc
    
def remove_padding(added_nl,added_nc,imgRec):
    nl_updated,nc_updated,_ = imgRec.shape
    imgRec = imgRec[0:nl_updated-added_nl,0:nc_updated-added_nc,:]
    return imgRec

def YCbCr(img):
    img = img.astype(np.float32)
    YCbCr_matrix = np.array([[0.299,0.587,0.114],[-0.168736,-0.331264,0.5],[0.5,-0.418688,-0.081312]])
    YCbCr_matrix_2 = np.array([0, 128, 128])
    img = np.dot(img, YCbCr_matrix.T) + YCbCr_matrix_2
    Y = img[:,:,0]
    Cb = img[:,:,1]
    Cr = img[:,:,2]
    return Y,Cb,Cr,YCbCr_matrix,YCbCr_matrix_2
    
def remove_YCbCr(img, YCbCr_matrix, YCbCr_matrix_2):
    img -= YCbCr_matrix_2
    remove_YCbCr_matrix = np.linalg.inv(YCbCr_matrix)
    img = np.dot(img, remove_YCbCr_matrix.T)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def encoder(img,cm_red,cm_green,cm_blue,cm_grey):
    img,added_nl,added_nc = add_padding(img)
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    showImg(img,"Imagem com padding")
    showImg(R,"Red",cm_red)
    showImg(G,"Green",cm_green)
    showImg(B,"Blue",cm_blue) 
    #print("Matriz R")  
    #showSubMatrix(R,8,8,8)
    Y,Cb,Cr,YCbCr_matrix,YCbCr_matrix_2 = YCbCr(img)
    showImg(Y,"Y",cm_grey)
    showImg(Cb,"Cb",cm_grey)
    showImg(Cr,"Cr",cm_grey)
    CbResize,CrResize = downsampling(Cb,Cr, 0.5, 1)
    showImg(CbResize,"Cb downsampling 4:2:2",cm_grey)
    showImg(CrResize,"Cr downsampling 4:2:2",cm_grey)
    #print("------------")
    #print("Matriz Y")
    #showSubMatrix(Y,8,8,8)
    #print("------------")
    #print("Matriz Cb")
    #showSubMatrix(Cb,8,8,8)
    return Y,Cb,Cr,added_nl,added_nc,YCbCr_matrix,YCbCr_matrix_2

def decoder(Y,Cb,Cr,added_nl,added_nc,YCbCr_matrix,YCbCr_matrix_2):
    imgRec = np.stack((Y, Cb, Cr), axis=-1)
    imgRec = remove_YCbCr(imgRec, YCbCr_matrix, YCbCr_matrix_2)
    imgRec = remove_padding(added_nl,added_nc,imgRec)
    return imgRec

def main():
    fName = "./imagens/airport.bmp"
    img = plt.imread(fName) 
    # print(img.shape)
    showImg(img,fName)
    
    cm_red = clr.LinearSegmentedColormap.from_list("red",[(0,0,0),(1,0,0)], N=256)
    cm_green = clr.LinearSegmentedColormap.from_list("green",[(0,0,0),(0,1,0)], N=256)
    cm_blue = clr.LinearSegmentedColormap.from_list("blue",[(0,0,0),(0,0,1)], N=256)
    cm_grey = clr.LinearSegmentedColormap.from_list("grey",[(0,0,0),(1,1,1)], N=256)
    
    Y,Cb,Cr,added_nl,added_nc,YCbCr_matrix,YCbCr_matrix_2 = encoder(img,cm_red,cm_green,cm_blue,cm_grey)
    imgRec = decoder(Y,Cb,Cr,added_nl,added_nc,YCbCr_matrix,YCbCr_matrix_2)
    showImg(imgRec,"Imagem Reconstruida / sem padding")

if __name__ == "__main__":
    main()