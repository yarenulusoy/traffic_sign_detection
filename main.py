import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
keras.optimizers.Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model #eklendi


path = "trafficData"  # tum resimlerin ve sınıfların bulundugu klasor
labelFile = 'labels.csv'  #sınıf adlarını iceren excel dosyası
batch_size_val = 50
steps_per_epoch_val = 2000
epochs_val = 10 #tur sayisi modelin kaç kere ileri ve geri yayılım yapacağı
imageDimesions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2
#her klasörü tek tek okuyarak kaç sınıf oldugunu tespit eder ve hepsini bir matrise atar
#daha sonra her klasoru tek tek alır

count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Toplam sinif sayisi:", len(myList))
noOfClasses = len(myList)
print("Siniflari import et...")
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        # curImg = cv2.resize(curImg,(32,32))
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

#verileri bol
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
steps_per_epoch_val = len(X_train)//batch_size_val #eklendi
validation_steps = len(X_test)//batch_size_val #eklendi


#goruntu sayisi ile etiket sayisinin eşleşmesi kontrolu hata mesajlari boyutlar uygun mu kontrol ediyor
print("Data Shapes")
print("Train", end="");
print(X_train.shape, y_train.shape)
print("Validation", end="");
print(X_validation.shape, y_validation.shape)
print("Test", end="");
print(X_test.shape, y_test.shape)
assert (X_train.shape[0] == y_train.shape[
    0]), "Egitim setindeki etiket sayisina esit olmayan resim sayisi"
assert (X_validation.shape[0] == y_validation.shape[
    0]), "Dogrulama kumesindeki etiket sayisina esit olmayan resim sayisi"
assert (X_test.shape[0] == y_test.shape[0]), "Test setindeki etiket sayisina esit olmayan goruntu sayisi"
assert (X_train.shape[1:] == (imageDimesions)), " Egitim goruntulerinin boyutlari yanlis"
assert (X_validation.shape[1:] == (imageDimesions)), "Dogrulama resimlerinin boyutlari yanlis"
assert (X_test.shape[1:] == (imageDimesions)), "Test goruntulerinin boyutlari yanlis"

#excel dosyasini okuma
data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))

#Tum siniflardan ornek goruntuler
num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            num_of_samples.append(len(x_selected))

#Her kategori icin sayı semasi
print(num_of_samples)
plt.figure(figsize=(12, 4)) #degisen
plt.bar(range(0, num_classes), num_of_samples)
plt.xlabel("Sinif sayisi")
plt.ylabel("Goruntu sayisi")
plt.show()


#Goruntuleri isle

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)  #Resimleri Grayscale donustur
    img = equalize(img)  #Resimleri aydınlat
    img = img / 255  #Degerleri normallestirmek
    return img


X_train = np.array(list(map(preprocessing, X_train)))  #Goruntuleri etiketlemek ve on isleme yapmak
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))
cv2.imshow("GrayScale Images",
           X_train[random.randint(0, len(X_train) - 1)])  #Egitimin dogru yapildigini kontrol et

#Derinlik ekle
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

#Goruntulerin artirilmasi
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             # 0.1 = 10%
                             height_shift_range=0.1,
                             zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                             shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                             rotation_range=10)  # DEGREES
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train,
                       batch_size=20)
X_batch, y_batch = next(batches)

#Artirilmis goruntu orneklerini gostermek icin
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0], imageDimesions[1]))
    axs[i].axis('off')
plt.show()

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)


#Yapay sinir agi modeli
def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)
    #Her kenardan 2 piksel kaldirir
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)  # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING
    no_Of_Nodes = 500  # NO. OF NODES IN HIDDEN LAYERS
    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1),
                      activation='relu')))  #Daha fazla donusum katmanlari ekleme
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))  #Filtrelerin derinligini etkilemez

    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))  # OUTPUT LAYER
    # COMPILE MODEL
    model.compile(keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#TRAIN
model = myModel()
print(model.summary())
history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                              steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                              validation_data=(X_validation, y_validation), shuffle=1)

#Grafik
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

#Modeli kaydetme
#pickle_out = open("model_trained.p", "wb")  # wb = WRITE BYTE
#pickle.dump(model, pickle_out)
model.save('traffic_model.h5')  # creates a HDF5 file 'my_model.h5' #eklendi
model1 = load_model('traffic_model.h5')#eklendi
#pickle_out.close()
cv2.waitKey(0)