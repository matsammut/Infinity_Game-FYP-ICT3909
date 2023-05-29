import cv2 as cv
import numpy as np
import random
from BestFace import *
import torch
from DBSCAN import *

def main(path, num_screenshots, scale, AI, key):

    capture = cv.VideoCapture(path)
    success, frame = capture.read()

    # fps = capture.get(cv.CAP_PROP_FPS)
    # vid_length = 30
    total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    # total_frames = vid_length * 60 * fps

    # Number of frames to take from the film
    # num_screenshots = 1500
    desired_frames = np.sort(random.sample(range(int(total_frames)), num_screenshots))
    # Take a frame per interval
    # frame_interval = 12
    # desired_frames = frame_interval * np.arange(total_frames)

    backgrounds = []
    bgCounter = 0
    bgSpacing = 26
    bgDir = "Film/BackGrounds"
    empty_dir(bgDir)
    frameDir = "Film/Test"
    empty_dir(frameDir)

    for i in desired_frames:
        capture.set(1, i-1)
        success, frame = capture.read(1)
        frameId = capture.get(1)
        # scale = 1
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)

        dimensions = (width, height)

        frame = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
        frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        haar_face = cv.CascadeClassifier('haar_face.xml')



        # Get Faces
        faces_rect = haar_face.detectMultiScale(frameGray, scaleFactor=1.1, minNeighbors=6)
        face_counter = 0
        for (x, y, w, h) in faces_rect:
            face = frame[y:y+h,x:x+w]
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv.imwrite("Film/Test/frame%d(%d).jpg" % (frameId, face_counter), face)
            face_counter += 1

        # Get backgrounds
        if extract_backgrounds(frame) and bgCounter < 20 and bgSpacing > 25 and face_counter == 0:
            cv.imwrite("Film/BackGrounds/BG%d.jpg" % bgCounter, frame)
            bgCounter += 1
            bgSpacing = 0
        bgSpacing += 1

        # print(f'Number of faces ={len(faces_rect)}')
        cv.imshow('Video', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()
    DBSCAN()
    empty_dir("Sil")
    model(AI, path, key)
    music_shazam(path)
    import_all()

# path = "Film/Films/Django.mp4"
path = input("Please enter the path for the film")
grabs = int(input("Please enter the number of screen grabs the program should take\n"))
scale = float(input("Please enter the Scale you wish to analyse the video at\n"))
AI = input("Do you wish to generate additional assets using AI art? y/n (Note: requires a valid Open AI key)\n")
key = 0
if AI == 'y':
    key = input("Please enter your key\n")
    AI = True
else:
    AI = False
main(path, grabs, scale, AI, key)


