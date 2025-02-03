import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np

def showImg(img,title,cmap=None):
    plt.figure()
    plt.imshow(img,cmap)
    plt.axis("off")
    plt.title(title)
    plt.show()


def encoder(img,cm_red,cm_green,cm_blue):
    # 3 canais da imagem
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    showImg(R,"Red",cm_red)
    showImg(G,"Green",cm_green)
    showImg(B,"Blue",cm_blue)

def decoder(img):
    pass

def main():
    fName = "./Multimedia/TP1/imagens/airport.bmp"
    img = plt.imread(fName) 
    showImg(img,fName,None)
    
    print(type(img))
    print(img.shape)
    print(img[0:8,0:8,0])
    print(img.dtype)
    
    cm_red = clr.LinearSegmentedColormap.from_list("red",[(0,0,0),(1,0,0)], N=256)
    cm_green = clr.LinearSegmentedColormap.from_list("green",[(0,0,0),(0,1,0)], N=256)
    cm_blue = clr.LinearSegmentedColormap.from_list("blue",[(0,0,0),(0,0,1)], N=256)
    
    encoder(img,cm_red,cm_green,cm_blue)
    ###############
    decoder(img)

if __name__ == "__main__":
    main()