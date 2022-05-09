from PIL import Image, ImageEnhance
from numpy import asarray
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd

def RecognitionImage(name, clf):
    #input the image
    im = Image.open(str(name)).convert("L")

    #crop to make it a square
    w, h = im.size
    if(w>h):
        cut = (w-h)//2
        img = im.crop((cut,h,(w-cut),0))
    elif(w<h):
        cut = (h-w)//2
        img = im.crop((0, cut ,w,(h-cut)))
    else:
        img=im

    #compress this square image to 28*28 pixel image
    Max = (28,28)
    img.thumbnail(Max)

    #convert image into 2D-array of pixels
    data = asarray(img)
    leng, bre = (data.shape)

    #Covert 2D-array to 1D-array and save as csv file
    arr =np.array([])
    arr2 = np.array([x for x in range(0,784,1)])
    for x in range(leng):
        for y in range(bre):
            arr = np.append(arr, (255-data[x][y]))
            
    combi = np.vstack((arr2,arr))
    np.savetxt('array.csv',combi,delimiter=',', fmt='%d')

    #Predict
    made = pd.read_csv('array.csv').values
    return clf.predict([made[0]])[0]
