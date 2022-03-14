import os
from pickletools import uint8
from PIL import Image
import numpy as np
import cv2 as cv
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recogniser = cv.face.LBPHFaceRecognizer_create()



def trainer():
    current_id = 0
    label_ids = {}
    x_train = []
    y_labels = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = str(os.path.basename(root))   #root can be replaced with os.path.dirname(path)
                #print(label, path)

                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                #print(label_ids)

                # x_train.append(path)  {verify img, turn into numpy array, gray}
                # y_labels.append(path)   {some number}
                pil_image = Image.open(path).convert("L")#conv to grey
                size = (550,550)
                final_image = pil_image.resize(size, Image.ANTIALIAS)

                image_array = np.array(pil_image, "uint8")#conv img to numbers
                #print(image_array)

                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5) #face detector

                for (x,y,w,h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    #print(y_labels)
    #print(x_train)

    with open("labels2.pickle", 'wb') as f:
        pickle.dump(label_ids, f)

    recogniser.train(x_train, np.array(y_labels))
    recogniser.save("trainer2.yml")

