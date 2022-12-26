import numpy as np
import cv2
import pickle
from keras.models import load_model

#Kamera ayarlari
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

#Kamerayi kur
capture = cv2.VideoCapture(0)
capture.set(3, frameWidth)
capture.set(4, frameHeight)
capture.set(10, brightness)
#Modeli import et
pickle_in = open("traffic_model.h5", "rb")  ## rb = READ BYTE
#model = pickle.load(pickle_in)
model= load_model('traffic_model.h5') #eklendi


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getCalssName(classNo):
    if classNo == 0:
        return 'Hiz siniri  20 km/h'
    elif classNo == 1:
        return 'Hiz siniri 30 km/h'
    elif classNo == 2:
        return 'Hiz siniri 50 km/h'
    elif classNo == 3:
        return 'Hiz siniri limit 60 km/h'
    elif classNo == 4:
        return 'Hiz siniri 70 km/h'
    elif classNo == 5:
        return 'Hiz siniri 80 km/h'
    elif classNo == 6:
        return 'Hiz sinirlamasi sonu 80 km/h'
    elif classNo == 7:
        return 'Hiz siniri 100 km/h'
    elif classNo == 8:
        return 'Hiz siniri 120 km/h'
    elif classNo == 9:
        return 'Gecis yok'
    elif classNo == 10:
        return '3,5 tonun uzerindeki araclara gecis yasak'
    elif classNo == 11:
        return 'Sonraki kavsakta gecis hakki'
    elif classNo == 12:
        return 'Oncelikli yol'
    elif classNo == 13:
        return 'Yol ver'
    elif classNo == 14:
        return 'Dur'
    elif classNo == 15:
        return 'Arac yok'
    elif classNo == 16:
        return '3.5 tonun uzerindeki araclar yasakli'
    elif classNo == 17:
        return 'Giris yok'
    elif classNo == 18:
        return 'Genel uyari'
    elif classNo == 19:
        return 'Soldaki tehlikeli viraj'
    elif classNo == 20:
        return 'Sagdaki tehlikeli viraj'
    elif classNo == 21:
        return 'Cift egri'
    elif classNo == 22:
        return 'Engebeli yol'
    elif classNo == 23:
        return 'Kaygan yol'
    elif classNo == 24:
        return 'Sagda yol daraliyor'
    elif classNo == 25:
        return 'Yol calismasi'
    elif classNo == 26:
        return 'Trafik sinyalleri'
    elif classNo == 27:
        return 'Yaya gecidi'
    elif classNo == 28:
        return 'Cocuk gecidi'
    elif classNo == 29:
        return 'Bisiklet yolu'
    elif classNo == 30:
        return 'Buza dikkat edin'
    elif classNo == 31:
        return 'Vahsi hayvan gecebilir'
    elif classNo == 32:
        return 'Tum hiz ve gecis limitlerinin sonu'
    elif classNo == 33:
        return 'Saga don'
    elif classNo == 34:
        return 'Sola don'
    elif classNo == 35:
        return 'Sadece ileri'
    elif classNo == 36:
        return 'Duz veya saga git'
    elif classNo == 37:
        return 'Duz veya sola git'
    elif classNo == 38:
        return 'Sagdan gidiniz'
    elif classNo == 39:
        return 'Soldan gidiniz'
    elif classNo == 40:
        return 'Dolambac zorunlu'
    elif classNo == 41:
        return 'Cikmaz Yol'
    elif classNo == 42:
        return '3.5 tonun uzerindeki araclarin gecisine son verilmesi'


while True:

    #Grountuleri oku
    success, imgOrignal = capture.read()

    #Goruntuleri işle
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "  ADI:", (0, 35), font, 0.75, (155, 155, 0), 2, cv2.LINE_AA)
    # TAHMİN
    predictions = model.predict(img)
    predict_x = model.predict(img)  # eklendi
    classIndex = np.argmax(predict_x, axis=1)  # eklendi
    probabilityValue = np.amax(predictions)
    if probabilityValue >threshold:
        #print(getCalssName(classIndex))
        cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (100, 35), font, 0.75, (155, 155, 0), 2,cv2.LINE_AA)
        cv2.putText(imgOrignal,"Tahmin: %"+ str(round(probabilityValue * 100, 2)), (20, 75), font, 0.75, (155, 155, 0), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break