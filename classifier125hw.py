from classifier125 import Xtrainscale
import numpy as np
from numpy.core.fromnumeric import resize
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps


X= np.load("image.npz")["arr_0"]
y= pd.read_csv("labels.csv")["labels"]

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y, test_size=500, train_size=3500, random_state=9)

Xtrainscale= Xtrain/255.0
Xtestscale= Xtest/255.0

classifier = LogisticRegression(solver='saga', multi_class= 'multinomial').fit(Xtrainscale, Ytrain)

def predictletter(image):
    im_pil = Image.open(image).convert('L')
    im_resize= im_pil.resize((28,28), Image.ANTIALIAS)
    pixel_filter=20
    min_pixel = np.percentile(im_resize, pixel_filter)
    im_resize_inverted= np.clip(255-im_resize, 0, 255)
    max_pixel = np.max(im_resize)
    im_resize_inverted = np.asarray(im_resize_inverted)/max_pixel
    test_sample = np.array(im_resize_inverted).reshape(1,784)
    prediction = classifier.predict(test_sample)
    return prediction[0]

