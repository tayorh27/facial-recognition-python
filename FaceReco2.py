
import cv2 as cv
import PySimpleGUI as sg
import numpy as np
import pickle
import os
import facetrain
import time

def main():

    start_time = time.time()

    def checkExist(path):
        isExist = os.path.exists(path)

        if isExist:
            return 1
        else:
            return 0
   
    sg.theme("LightBlue")
    
    #layout for main window
    lout = [
        [sg.Text("Keep within camera range...")],
        #[sg.Frame("Frame",[[sg.T(s=15)]], size = (550,550), key="Capture")]
        [sg.Image(filename='', size = (450,450), key='-Capture-')],
        [sg.Button("New User?", key="-NewUserBtn-", visible = False)],
        [sg.Text(text = "Enter your name", key = "-NewUserMsg-", text_color="Black", visible = False), sg.Text("Name already taken.", key = "taken", visible = False)],
        [sg.Input(s=25, key = "-Newname-", focus=True, visible=False)],
        [sg.Button(button_text="OK", key = "submitname", visible=False)],
        [sg.Text(text = "Collecting face data...",key="collect", visible = False), sg.Text(text="Complete!", key = "complete", visible = False)]
    ]
    

    face_cascade = cv.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')#accesses classifier
    recogniser = cv.face.LBPHFaceRecognizer_create()#creates recogniser
    recogniser.read("trainer2.yml")#uses the YAML file to recognise user's face

    facetrain.trainer()


    labels = {} #creates labels dict

    with open("labels2.pickle", 'rb') as f:            #loads the labels from pickle into dict
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}     

    win = sg.Window(title = "Face Recogniser", layout = lout, location=(100,100))

    cap = cv.VideoCapture(0)

    while True:        

        event, values = win.read(timeout=20)
        if event=="Exit" or event==sg.WIN_CLOSED:
            break

        #opens window to create new user
        if event == "-NewUserBtn-":
            win["-NewUserMsg-"].update(visible = True)
            win["-Newname-"].update(visible = True)
            win["submitname"].update(visible = True)
            
        if event == "submitname":
            count=0
            nameID = values["-Newname-"]
            path = "Images/" + nameID
            if checkExist(path) == 1:
                win["taken"].update(visible = True)
                continue
            elif checkExist(path) == 0:
                os.makedirs(path)
                win["taken"].update(visible = False)
                win["submitname"].update(disabled = True)

            while count<50:#inside for loop
                name = "./Images/" + nameID + "/" + str(count) + ".jpg"
                if faces is not None:
                    print("Creating images....." + name)
                    cv.imwrite(name, mframe)
                count += 1
            win["complete"].update(visible = True)
            win["-NewUserBtn-"].update(visible = False)
            facetrain.trainer()


        ret, mframe = cap.read() #reads the image at the moment

        gray = cv.cvtColor(mframe, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for(x,y,w,h) in faces:
            #print(x,y,w,h)  #region of interest (the face when it shows up)
            roi_gray = gray[y:y+h, x:x+w]
            roi_colour = mframe[y:y+h, x:x+w]

            #draw a rectangle around face detected
            color = (255, 0, 255)
            stroke = 2
            width = x + w #end coordinates for x
            height = y + h #end coordinates for y
            cv.rectangle(mframe, (x,y), (width, height), color, stroke)
                    
            #to recognise the user
            id_, conf = recogniser.predict(roi_gray)
            print(conf)
            if conf >= 99:
                #shows user's name if confidence over certain level
                font = cv.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color=(255, 255,255)
                stroke = 2
                cv.rectangle(mframe, (x, y+h), (x+w,y+h+20), (255,0,255), cv.FILLED)
                cv.putText(mframe, name, (x,y+h+10), font, 0.5, color, stroke, cv.LINE_AA)
                continue

            else:
                font = cv.FONT_HERSHEY_SIMPLEX
                color=(255, 255,255)
                stroke = 2
                #cv.rectangle(mframe, (x, y+h), (x+w,y+h+20), (255,0,255), cv.FILLED)
                #cv.putText(mframe, "?", (x,y+h+10), font, 0.5, color, stroke, cv.LINE_AA)
                if time.time() - start_time >= 10:
                    win["-NewUserBtn-"].update(visible = True)
                
    
        #constantly updates the image element of the window with images read from the camera.   
        imgbytes = cv.imencode('.png', mframe)[1].tobytes()
        win['-Capture-'].update(data=imgbytes)

        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    win.close()

main()
