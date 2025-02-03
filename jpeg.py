import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np

def showImg(img,title,cmap=None):
    plt.figure()
    plt.imshow(img,cmap)
    plt.axis("off")
    plt.title(title)
    plt.show()


def showSubMatrix(img,i,j,dim):
    nd = img.ndim # numero de dimensoes da matriz
    if nd==2:
        print(img[i:i+dim,j:j+dim])
    elif nd==3:
        print(img[i:i+dim,j:j+dim,0])

def encoder(img,cm_red,cm_green,cm_blue,cm_grey):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    Grey = img[:,:,:]
    showImg(R,"Red",cm_red)
    showImg(G,"Green",cm_green)
    showImg(B,"Blue",cm_blue)
    showImg(Grey,"Grey",cm_grey)
    return R,G,B

def decoder(R,G,B):
    nl,nc = R.shape # devolve numero de linhas e colunas
    imgRec = np.zeros((nl,nc,3),dtype=np.uint8)
    imgRec[:,:,0] = R
    imgRec[:,:,1] = G
    imgRec[:,:,2] = B
    return imgRec

def main():
    fName = "./TP1/imagens/airport.bmp"
    img = plt.imread(fName) 
    showImg(img,fName)
    
    cm_red = clr.LinearSegmentedColormap.from_list("red",[(0,0,0),(1,0,0)], N=256)
    cm_green = clr.LinearSegmentedColormap.from_list("green",[(0,0,0),(0,1,0)], N=256)
    cm_blue = clr.LinearSegmentedColormap.from_list("blue",[(0,0,0),(0,0,1)], N=256)
    cm_grey = clr.LinearSegmentedColormap.from_list("grey",[(0,0,0),(1,1,1)], N=256)
    
    R,G,B = encoder(img,cm_red,cm_green,cm_blue,cm_grey)
    ###############
    imgRec = decoder(R,G,B)
    showImg(imgRec,"Imagem Reconstruida")
if __name__ == "__main__":
    main()