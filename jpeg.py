import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np


def encoder():
    pass

def decoder():
    pass

def main():
    fName = "./Multimedia/TP1/imagens/airport.bmp"
    img = plt.imread(fName) 
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.title(fName)
    plt.show()
    print(type(img))
    print(img.shape)

if __name__ == "__main__":
    main()