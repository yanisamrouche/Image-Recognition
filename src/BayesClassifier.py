import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#chemin vers le répertoire des données
datadir = "C:/Users/admin/PycharmProjects/MachineLearning/src/Data"

#les differentes categories de notre echantillon
categories = ["Mer", "Ailleurs"]

img_size = 500
training_data = []


def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category) # chemin vers Mer ou Ailleurs
        class_num = categories.index(category) # donner un num pour nos classes {Mer : +1, Ailleurs : -1}
        if class_num == 0:
            class_num = +1
        else:
            class_num = -1
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_GRAYSCALE
            features_array = cv2.resize(img_array, (img_size, img_size))
            training_data.append([features_array, class_num])

create_training_data()

#mélanger les données
#random.shuffle(training_data)
#for s in training_data: print(s[1])

X = [] #features sets
y = [] #label (class) sets

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1) #1


#pour sauvegarder les données
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickel_in = open("X.pickle", "rb")
X = pickle.load(pickel_in)

#print("L'image appartient a la classe : ",y[1])
#plt.imshow(X[1])
#plt.show()

#redemensionner les features
nsamples, nx, ny, nz = X.shape
d2_X = X.reshape((nsamples,nx*ny*nz))

#apprentissage
#decoupper l'echantillon en partie train/test
X_train, X_test, y_train, y_test = train_test_split(d2_X, y, test_size=0.20)

classifieur = GaussianNB()
classifieur.fit(X_train,y_train)
y_predits = classifieur.predict(X_test)

print("Les vraies classes :")
print(y_test)
print("Les classes prédites :")
print(y_predits)

from sklearn.metrics import accuracy_score
print("le taux de réussite avec le classifieur de Bayes : ",accuracy_score(y_test,y_predits))
