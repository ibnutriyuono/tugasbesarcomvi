import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def load_images(folder):
    images = np.array([])
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        histogram, bin_edges = np.histogram(gray, bins=10, range=(0, 256))
        if histogram is not None:
            images = np.append([[images]], histogram)
    images = images.reshape(int(len(images)/10), 10)
    return images


imgtrain = load_images('./Apple Braeburn')
imgtrain2 = load_images('./Apple Golden 1')
imgtrain3 = load_images('./Apple Granny Smith')
imgtrain4 = load_images('./Apple Golden 2')
imgtrain5 = load_images('./Apple Golden 3')
target = []
for i in range(len(imgtrain)):
    target.append(0)
for i in range(len(imgtrain2)):
    target.append(1)
for i in range(len(imgtrain3)):
    target.append(2)
for i in range(len(imgtrain4)):
    target.append(3)
for i in range(len(imgtrain5)):
    target.append(4)

train = np.concatenate((imgtrain, imgtrain2, imgtrain3, imgtrain4, imgtrain5))
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train, target)

test_0 = load_images('./test/Apple Braeburn')
test_1 = load_images('./test/Apple Golden 1')
test_2 = load_images('./test/Apple Granny Smith')
test_3 = load_images('./test/Apple Golden 2')
test_4 = load_images('./test/Apple Golden 3')
test = np.concatenate((test_0, test_1, test_2, test_3, test_4))

y_pred = neigh.predict(test)
y_test = []
for i in range(len(test_0)):
    y_test.append(0)
for i in range(len(test_1)):
    y_test.append(1)
for i in range(len(test_2)):
    y_test.append(2)
for i in range(len(test_3)):
    y_test.append(3)
for i in range(len(test_4)):
    y_test.append(4)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
