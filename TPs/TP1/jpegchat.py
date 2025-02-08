import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np

def showImg(img, title, cmap=None):
    plt.figure()
    plt.imshow(img, cmap)
    plt.axis("off")
    plt.title(title)
    plt.show()

def add_padding(img):
    nl, nc, _ = img.shape
    resto_nl = nl % 32
    resto_nc = nc % 32
    if resto_nl != 0:
        add_nl = 32 - resto_nl
        last_line = img[-1:, :, :]
        array_add_nl = np.repeat(last_line, add_nl, axis=0)
        img = np.vstack((img, array_add_nl))
    if resto_nc != 0:
        add_nc = 32 - resto_nc
        last_column = img[:, -1:, :]
        array_add_nc = np.repeat(last_column, add_nc, axis=1)
        img = np.hstack((img, array_add_nc))
    return img, add_nl, add_nc

def remove_padding(added_nl, added_nc, imgRec):
    nl_updated, nc_updated, _ = imgRec.shape
    return imgRec[:nl_updated - added_nl, :nc_updated - added_nc, :]

def YCbCr(img):
    img = img.astype(np.float32) / 255.0  # Normaliza para float entre 0 e 1
    YCbCr_matrix = np.array([[0.299, 0.587, 0.114],
                              [-0.168736, -0.331264, 0.5],
                              [0.5, -0.418688, -0.081312]])
    YCbCr_matrix_2 = np.array([0, 128, 128])
    
    img_ycbcr = np.dot(img, YCbCr_matrix.T) * 255 + YCbCr_matrix_2
    return img_ycbcr[:, :, 0], img_ycbcr[:, :, 1], img_ycbcr[:, :, 2], YCbCr_matrix, YCbCr_matrix_2

def remove_YCbCr(img, YCbCr_matrix, YCbCr_matrix_2):
    img = img.astype(np.float32)
    remove_YCbCr_matrix = np.linalg.inv(YCbCr_matrix)
    img_rgb = (img - YCbCr_matrix_2) @ remove_YCbCr_matrix.T / 255.0
    return np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)

def encoder(img, cm_red, cm_green, cm_blue, cm_grey):
    img, added_nl, added_nc = add_padding(img)
    Y, Cb, Cr, YCbCr_matrix, YCbCr_matrix_2 = YCbCr(img)
    
    showImg(img, "Imagem com padding")
    showImg(Y, "Y", cm_grey)
    showImg(Cb, "Cb", cm_grey)
    showImg(Cr, "Cr", cm_grey)
    
    return Y, Cb, Cr, added_nl, added_nc, YCbCr_matrix, YCbCr_matrix_2

def decoder(Y, Cb, Cr, added_nl, added_nc, YCbCr_matrix, YCbCr_matrix_2):
    imgRec = np.stack((Y, Cb, Cr), axis=-1)
    imgRec = remove_YCbCr(imgRec, YCbCr_matrix, YCbCr_matrix_2)
    return remove_padding(added_nl, added_nc, imgRec)

def main():
    fName = "./imagens/airport.bmp"
    img = plt.imread(fName)
    showImg(img, fName)
    
    cm_grey = clr.LinearSegmentedColormap.from_list("grey", [(0,0,0), (1,1,1)], N=256)
    
    Y, Cb, Cr, added_nl, added_nc, YCbCr_matrix, YCbCr_matrix_2 = encoder(img, None, None, None, cm_grey)
    imgRec = decoder(Y, Cb, Cr, added_nl, added_nc, YCbCr_matrix, YCbCr_matrix_2)
    showImg(imgRec, "Imagem Reconstruída / sem padding")

if __name__ == "__main__":
    main()
