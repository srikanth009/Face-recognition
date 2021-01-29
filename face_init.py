#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:48:40 2020

@author: basu
"""
from PIL import Image
import face_recognition
import cv2
import numpy as np
import os
import datetime
import telegram
os.chdir("/home/basu/projects/identification")


def preprocess():
    filet = os.listdir("unknown_log/")
    for i in range(len(filet)):
        im = Image.open('unknown_log/'+filet[i])
        k = 'unknown_log/'+filet[i][:-3]+"png"
        im.save(k)

def send(selection, msg):
    token="*******"
    chatid="****"
    chatid2 = "*****"
    bot = telegram.Bot(token=token)
    if(selection==0):
        bot.sendMessage(chat_id=chatid2, text=msg)
    else:
        filex = os.listdir("unknown_log/")
        preprocess()
        for i in range(len(filex)):
            bot.send_photo(chat_id=chatid2,photo = open("unknown_log/"+filex[i],"rb"))
            bot.sendMessage(chat_id=chatid2,text=filex[i][:-4])

def last_line(name):
    with open('data.txt', 'r') as f:
        lines = f.read().splitlines()
        if(len(lines)!=0):
            lastline = lines[-1]
        else:
            return 1
    lastline = lastline.split(" ")
    if(lastline[0]==name):
        return 0
    else:
        return 1

def notify(name):
    file1 = open(r"data.txt","a+")
    dfg = str(datetime.datetime.now())
    str1 = list(name + " found :: " + dfg[0:16] + "\n")
    if(name=="Unknown"):
        send(0,"".join(str1))
    print("".join(str1),"\n")
    file1.writelines(str1)


def identify():
    video_capture = cv2.VideoCapture(0)
    inp = os.listdir("face_data/")
    known_face_names = []
    im = []
    known_face_encodings = []
    for i in range(len(inp)):
        known_face_names.append(str(inp[i]))
        im.append(face_recognition.load_image_file("face_data/"+inp[i]))
        known_face_encodings.append(face_recognition.face_encodings(im[i])[0])
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    i=0
    while True:
        ret, frame = video_capture.read()
        i = i+1
        if(i%7==0):
            rgb_small_frame = frame[:, :, ::-1]
            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    face_names.append(name)
            process_this_frame = not process_this_frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 1
                right *= 1
                bottom *= 1
                left *= 1

                if(name!="Unknown"):
                    notify(name)
                    pass
                else:
                    strg = "unknown_log/"+"Unknown"+str(datetime.datetime.now())+".jpg"
                    cv2.imwrite(strg,frame)
                    notify(name)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow("Video",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()












