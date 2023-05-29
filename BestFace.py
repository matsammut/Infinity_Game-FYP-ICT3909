import os
import cv2
import shutil
import numpy as np
import tensorflow as tf
import keras
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model
import torch
import torchvision.transforms as transforms
from PIL import Image
import openpifpaf
import matplotlib.pyplot as plt
import urllib.request
from rembg import remove
import openai
import requests
from io import BytesIO
import moviepy.editor as mp
import shazamio
import asyncio


def remove_background(frame, resize):
    try:
        output = remove(frame)
    except:
        return False
    cv2.imwrite("body_nobg.png", output)

    image = Image.open("body_nobg.png")
    width, height = image.size
    left, top, right, bottom = width, height, 0, 0
    for x in range(width):
        for y in range(height):
            alpha = image.getpixel((x, y))[3]
            if alpha > 0:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    # Crop the image to the non-transparent region
    cropped_image = image.crop((left, top, right + 1, bottom + 1))
    resized_image = resize_img(cropped_image, 2)
    # bbox = image.getbbox()
    # cropped_image = image.crop(bbox)

    resized_image.save("test.png")
    return resized_image


def select_best_face(folder_path):
    best_face = None
    best_score = float('inf')
    min_size = 10
    faces = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)

            height, width, _ = image.shape
            if width >= height and width > min_size:
                new_width = min_size
                new_height = int(height * (min_size / width))
            elif height > width and height > min_size:
                new_height = min_size
                new_width = int(width * (min_size / height))
            else:
                # Skip the image if it is too small
                continue

            # Resize image to a fixed size for easier comparison
            target_size = (256, 256)  # You can adjust this to your desired size
            image_resized = cv2.resize(image, (new_width, new_height))

            # Calculate score based on least occlusion, least blur, and neutral emotion
            score = (cv2.Laplacian(image_resized, cv2.CV_64F).var() +
                     cv2.Laplacian(cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() +
                     abs(cv2.Laplacian(cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY), cv2.CV_64F).mean() - 50))
            tuple_score = (score, file_path)
            faces.append(tuple_score)
            if score < best_score:
                best_face = file_path
                best_score = score
    best_faces = sorted(faces, key=lambda x: x[0], reverse=True)
    return best_face, best_faces


def extract_backgrounds(frame):
    model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # # Check if the frame contains any main characters
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    resized_frame = cv2.resize(frame, (224, 224))
    input_frame = tf.keras.utils.array_to_img(resized_frame)
    array = np.array(input_frame)
    input_frame = preprocess_input(array)
    input_frame = np.expand_dims(input_frame, axis=0)

    # Check if there are any people in the foreground
    detections = model.predict(input_frame)
    class_indices = np.argmax(detections[0, :, :, 4:], axis=-1)
    # How confident is it that it is detecting a person
    confidence_scores = np.max(detections[0, :, :, 4:], axis=-1)
    # 15 being a person
    person_indices = np.where((class_indices == 15) & (confidence_scores > 0.2))[0]
    if len(person_indices) > 0:
        return False

    # Check if the frame is blurry
    laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    if laplacian_var < 100:
        return False

    # Add the frame to the backgrounds list
    return True


def empty_dir(dir):
    # Create a "Clusters" directory if it does not exist
    if not os.path.exists(dir):
        os.mkdir(dir)
    # Empty and Recreate the dir if it does
    if os.path.exists(dir):
        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir)


# not the purpose is not to have a perfect bounding box just a general idea of the body
# the idea is to reduce some of the noise for the pose detector
def character_bounding(face, frame):
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Detect face in full image
    face_cascade = cv2.CascadeClassifier('haar_face.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=6)

    # Find the face in the full image with the highest overlap
    best_face_overlap = 0
    best_face_box = None
    for face_box in faces:
        x, y, w, h = face_box
        face_roi = frame[y:y + h, x:x + w]
        # cv2.imshow("roi",face_roi)
        # cv2.waitKey(0)
        face_gray_resized = cv2.resize(face_gray, (w, h))
        overlap = np.sum((cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) == face_gray_resized)) / float(w * h)
        if overlap > best_face_overlap:
            best_face_overlap = overlap
            best_face_box = face_box

    # Check if a face was found
    if best_face_box is None:
        print("Face not found")
        return False
    else:
        # Find the body bounding box
        x, y, w, h = best_face_box

        body_height = int(h * 15)
        # because to give room to the top of the head for raised arms
        body_y = int(y - (h * 2))
        body_center = x + w // 2
        body_width = int(body_height * 0.3)
        body_x = int(body_center - body_width // 2)
        body_width = int(body_center + body_width // 2)
        body_box = (body_x, body_y, body_width, body_height)

        # Limit the bounding box to the size of the image
        if body_x < 0:
            body_width += body_x
            body_x = 0
        if body_y < 0:
            body_height += body_y
            body_y = 0
        if body_x + body_width > frame.shape[1]:
            body_width = frame.shape[1] - body_x
        if body_y + body_height > frame.shape[0]:
            body_height = frame.shape[0] - body_y

        frame = cv2.rectangle(frame, (body_x, body_y), (body_width, body_height), (0, 255, 0), 2)
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)
        cv2.imwrite("Bound_Check.png", frame)
        body = frame[body_y:body_y + body_height, body_x:body_width]
        # cv2.imshow("frame", body)
        # cv2.waitKey(0)
    return remove_background(body, 2)


def pose_detection(frame, pose_thresh):
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

    Body_parts = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4, "LShoulder": 5, "LElbow": 6,
                  "LWrist": 7, "RHip": 8, "RKnee": 9, "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
    Pose_pairs = [("Nose", "Neck"), ("Neck", "RShoulder"), ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
                  ("Neck", "LShoulder"), ("LShoulder", "LElbow"), ("LElbow", "LWrist"), ("Neck", "RHip"),
                  ("RHip", "RKnee"), ("RKnee", "RAnkle"), ("Neck", "LHip"), ("LHip", "LKnee"), ("LKnee", "LAnkle"),
                  ("Nose", "REye"), ("REye", "REar"), ("Nose", "LEye"), ("LEye", "LEar")]

    # Load the image and prepare it for inference
    height = frame.shape[0]
    width = frame.shape[1]
    thresh = 0.02

    # Set the input for the network and perform a forward pass
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    assert (len(Body_parts) == out.shape[1])

    # Create an empty array to store the detected keypoints
    keypoints = []

    for i in range(len(Body_parts)):
        # Get the heatmap for the current body part
        heatmap = out[0, i, :, :]

        # Find the location of the peak value in the heatmap
        _, conf, _, point = cv2.minMaxLoc(heatmap)

        # Scale the x and y coordinates to the size of the input image
        x = int(width * point[0] / out.shape[3])
        y = int(height * point[1] / out.shape[2])

        # Add the keypoint and its confidence score to the list
        keypoints.append((int(x), int(y)) if conf > thresh else None)
    print(keypoints)
    # Create a new image and draw the pose skeleton on it
    pose_image = frame.copy()
    for pair in Pose_pairs:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in Body_parts)
        assert (partTo in Body_parts)

        idFrom = Body_parts[partFrom]
        idTo = Body_parts[partTo]

        # Draw a line between the source and destination keypoints
        if keypoints[idFrom] and keypoints[idTo]:
            cv2.line(pose_image, keypoints[idFrom], keypoints[idTo], (0, 255, 0), 3)
            cv2.ellipse(pose_image, keypoints[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(pose_image, keypoints[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    num_keypoints = len(keypoints)
    num_present_keypoints = sum(keypoint is not None for keypoint in keypoints)

    # Calculate the percentage of present keypoints
    percentage_present = (num_present_keypoints / num_keypoints) * 100

    cv2.imwrite("pose.png", pose_image)
    if percentage_present >= pose_thresh:
        print("True")
        return True
    else:
        print("False")
        return False


def variations(path, path_save, key):
    openai.api_key = key
    prompt = openai.Image.create_variation(
        image=open(path, mode="rb"),
        n=1,
        size="512x512",
        # response_format="b64_json",
    )
    url = prompt['data'][0]['url']
    response = requests.get(url)

    img = Image.open(BytesIO(response.content))
    # print(url)
    # img.save("Generate/Test/Test.png")
    img.save(path_save)


def resize_img(img, resize):
    if resize == 1:
        resized_image = img.resize((256, 256))
    elif resize == 2:
        resized_image = img.resize((512, 512))
    elif resize == 3:
        resized_image = img.resize((1024, 1024))
    else:
        resized_image = img
    return resized_image


def model(use_variations, pathCap, key):
    path = "Clusters/"
    capture = cv2.VideoCapture(pathCap)
    dirs = os.listdir(path)
    model_counter = 0
    empty_dir("Models")
    for dir in dirs:
        folder = os.path.join(path, dir)
        best_faces = select_best_face(folder)[1]

        pose_thresh = 107
        while (pose_thresh > 55):
            pose_thresh -= 10
            print(pose_thresh)
            face_counter = 0
            done = False
            while (face_counter < len(best_faces)):
                face_path = best_faces[face_counter][1]
                face = cv2.imread(face_path)
                index = face_path.find("frame")
                frame = int(face_path[index + 5:-7])

                capture.set(1, frame)
                _, frame = capture.read()
                scale = 0.4
                width = int(frame.shape[1] * scale)
                height = int(frame.shape[0] * scale)

                dimensions = (width, height)

                frame = cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
                hold = character_bounding(face, frame)
                if hold == False:
                    face_counter += 1
                    continue
                pose = cv2.imread("test.png")
                # cv2.imshow("AfterBounding", pose)
                # cv2.waitKey(0)
                if pose_detection(pose, pose_thresh) == False:
                    face_counter += 1
                    continue
                done = True

                break

            if done:
                break
        if use_variations:
            if pose_thresh > 79:
                variations("test.png", "Generate/Test/Test.png", key)
                character_path = "Generate/Test/Test.png"
                shutil.copy("Generate/Test/Test.png", "pifuhd/sample_images")
                os.system('cd pifuhd && python -m apps.simple_test')
                model_path = "pifuhd/results/pifuhd_final/recon/result_test_512.obj"

                shutil.copy(model_path, "Models")
                rename = f"Models/Model{model_counter}.obj"
                model_counter += 1
                os.rename("Models/result_test_512.obj", rename)

        shutil.copy("test.png", "pifuhd/sample_images")
        shutil.copy("test.png", "Sil")
        renameSil = f"Sil/Silhouette{model_counter}.png"
        os.rename("Sil/test.png", renameSil)

        os.system('cd pifuhd && python -m apps.simple_test')
        model_path = "pifuhd/results/pifuhd_final/recon/result_test_512.obj"

        shutil.copy(model_path, "Models")
        rename = f"Models/Model{model_counter}.obj"
        model_counter += 1
        os.rename("Models/result_test_512.obj", rename)


async def recog_shazam(song):
    # video_path =  "Den.mp3"
    shazam = shazamio.Shazam()
    out = await shazam.recognize_song(song)
    # print(out)
    if len(out['matches']) > 0:
        return out['track']['title']
    else:
        # print(False)
        return False
    # tracks = await shazam.search_track(query='Lil', limit=5)
    # print(tracks)


def music_shazam(path):
    # Load the video and extract the audio
    video_path = path
    video_clip = mp.VideoFileClip(video_path)
    audio_clip = video_clip.audio

    duration = int(audio_clip.duration // 60) * 60
    audio_clip = audio_clip.subclip(0, duration)
    # Chop up the audio into 1-minute snippets
    snippet_duration = 30  # seconds
    snippets = [audio_clip.subclip(start, start + snippet_duration)
                for start in range(0, int(audio_clip.duration), snippet_duration)]
    counter = 0
    empty_dir("Music/Snippets")
    for snippet in snippets:
        snippet.write_audiofile(f"Music/Snippets/snippet_{counter}.mp3")
        counter += 1
    # Identify music tracks using Shazam

    shazam = shazamio.Shazam()
    music_snippets = {}
    snippet_array = []
    previous_music_track = ""
    counter = -1
    for i in range(len(snippets)):
        loop = asyncio.get_event_loop()
        current_snippet = f'snippet_{i}.mp3'
        song = loop.run_until_complete(recog_shazam(f"Music/Snippets/snippet_{i}.mp3"))
        print(f"Snippet {i}: {song}")
        if song == False:
            continue
        if previous_music_track == song:
            previous_music_track = song
            snippet_array.append(current_snippet)
        else:
            if counter != -1:
                music_snippets[counter] = snippet_array
            counter += 1
            snippet_array = []
            previous_music_track = song
            snippet_array.append(current_snippet)
    counter += 1
    music_snippets[counter] = snippet_array
    print(music_snippets)

    counter = 0
    empty_dir("Music/Tracks_Checking")
    for key in sorted(music_snippets.keys()):
        filenames = music_snippets[key]
        clips = []
        if len(filenames) > 1:
            for filename in filenames:
                clip = mp.AudioFileClip(os.path.join("Music/Snippets", filename))
                clips.append(clip)
            concatenated_clip = mp.concatenate_audioclips(clips)
            concatenated_clip.write_audiofile(f"Music/Tracks_Checking/music_track_{counter}.mp3")
            counter += 1

    # checking
    empty_dir("Music/Tracks")
    check_list = os.listdir("Music/Tracks_Checking")
    for tracks in check_list:
        loop = asyncio.get_event_loop()
        song = loop.run_until_complete(recog_shazam(os.path.join("Music/Tracks_Checking", tracks)))
        if song == False:
            os.remove(os.path.join("Music/Tracks_Checking", tracks))

    check_list = os.listdir("Music/Tracks_Checking")
    counter = 0
    for tracks in check_list:
        shutil.copyfile(os.path.join("Music/Tracks_Checking", tracks),
                        os.path.join("Music/Tracks", f"music_track_{counter}.mp3"))
        # os.rename(os.path.join("Music/Tracks",tracks),os.path.join("Music/Tracks", f"music_track_{counter}.mp3"))
        counter += 1


def import_all():
    models_source = 'Models'
    models_dest = 'Game\Thesis3D\Assets\Models'
    empty_dir(models_dest)

    for item in os.listdir(models_source):
        source = os.path.join(models_source, item)
        destination = os.path.join(models_dest, item)
        shutil.copy2(source, destination)

    bg_source = 'Film\BackGrounds'
    bg_dest = 'Game\Thesis3D\Assets\Backgrounds'
    empty_dir(bg_dest)

    for item in os.listdir(bg_source):
        source = os.path.join(bg_source, item)
        destination = os.path.join(bg_dest, item)
        shutil.copy2(source, destination)

    music_source = 'Music\Tracks'
    music_dest = 'Game\Thesis3D\Assets\Music'

    for item in os.listdir(music_source):
        source = os.path.join(music_source, item)
        destination = os.path.join(music_dest, item)
        shutil.copy2(source, destination)
