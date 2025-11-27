import cv2
import matplotlib.pyplot as plt
from Preprocess import process_image
import time
import os
import cv2
import pickle
from Image import Image
from glob import glob
import os
import cv2
from Preprocess import process_image
import time
import numpy as np
from threading import Thread

# get the start time
st = time.time()
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# get the dataset
query_folder_path = r"C:\Users\abedall\Desktop\MY_VERSIONS\myQuery"
#query_folder_path = r"C:\Users\abedall\Desktop\MY_VERSIONS\myQuery"
query = cv2.imread(query_folder_path + r"/" + 'DSC_2545.JPG')
query = process_image(query)
#cv2.imshow('query', query)
max_score = 0
max_score_id = -1
keypoint1, descriptor1 = sift.detectAndCompute(query, None)


def fetchKeypointFromFile(imagename):
    filepath = r"C:/Users/abedall/Desktop/graduiton project/dataset/features_extraxtion/keypoints/" + "DSC_" + str(imagename)+ "_kps" + ".json"
   # filepath = r"C:/Users/co90co/Desktop/graduaton project/dataset/features_extraxtion/keypoints/" + "DSC_" + str(imagename)+ "_kps" + ".json"

    keypoint = []
    if os.path.exists(filepath):
     file = open(filepath, 'rb')
     deserializedKeypoints = pickle.load(file)
     file.close()
     for point in deserializedKeypoints:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        keypoint.append(temp)
     return keypoint
    else:
        return 0


def fetchDescriptorFromFile(imagename):

    filepath = r"C:/Users/abedall/Desktop/graduiton project/dataset/features_extraxtion/descriptors/" + "DSC_" + str(imagename) + "_des" + ".json"
    #filepath = r"C:\Users\co90co\Desktop\graduation project\dataset\features_extraxtion\descriptors\\" + "DSC_" + str(imagename) + "_des" + ".json"
    if os.path.exists(filepath):
        file = open(filepath, 'rb')
        descriptor = pickle.load(file)
        file.close()

        return descriptor
    else:
        return False


def calculateResultsFor(image):
    global max_score
    global max_score_id

    descriptor2 = fetchDescriptorFromFile(image)
    if np.any(descriptor2) == [0]:
        matches = 0
        score = 0
        return score
    else:
        matches = calculateMatches(descriptor1, descriptor2)
        score = len((matches))
        if score > max_score:
            max_score = score
            max_score_id = image

        #return score



def calculateMatches(des1, des2):
    matches = bf.knnMatch(des1, des2, k=2)
    topResults = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            topResults.append([m])
    return topResults



i = 2332
threads = []



while i <= 3385:
    image= i
    temp = Thread(target=calculateResultsFor(image),args=(i))
    temp.start()
    temp.join()
    i=i+1


# for thread in threads:
#     thread.start()
#     thread.join()

print("the max score is : ", max_score, "|| with the id : ", max_score_id )

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')



cv2.waitKey(0)
cv2.destroyAllWindows()