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
import  numpy as np

# get the start time
st = time.time()
max_score = 0
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
#bf  = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
# get the dataset
query_folder_path = r"C:\Users\abedall\Desktop\MY_VERSIONS\myQuery"
query = cv2.imread(query_folder_path + r"/" + '66.jpg')
query = process_image(query)
keypoint1, descriptor1 = sift.detectAndCompute(query, None)

def fetchKeypointFromFile(imagename):
    filepath = r"C:/Users/abedall/Desktop/graduiton project/dataset/features_extraxtion/keypoints/" + "DSC_" + str(imagename)+ "_kps" + ".json"
    keypoint = []
    if os.path.exists(filepath):
     file = open(filepath, 'rb')
     deserializedKeypoints = pickle.load(file)
     file.close()
     for point in deserializedKeypoints:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        keypoint.append(temp)
     return keypoint
    else:return False


def fetchDescriptorFromFile(imagename):

    filepath = r"C:/Users/abedall/Desktop/graduiton project/dataset/features_extraxtion/descriptors/" + "DSC_" + str(imagename) + "_des" + ".json"
    if os.path.exists(filepath):
     file = open(filepath, 'rb')
     descriptor = pickle.load(file)
     file.close()
     return descriptor
    else:return False


def calculateResultsFor(image):
    descriptor2 = fetchDescriptorFromFile(image)
    if np.any(descriptor2) == [0]:
        matches = 0
        score = 0
        return score
    else:
        matches = calculateMatches(descriptor1, descriptor2)

        score = len((matches))
        return score

def calculateMatches(des1, des2):
    matches = bf.knnMatch(des1, des2,k=3)

    topResults = []
    for m, n ,v in matches:
        if m.distance < 0.6 * n.distance and m.distance< 0.6 * v.distance :
            topResults.append([m])
    return topResults

i = 2332
while i <= 3385 :
 image= i
 # print("image with number",image," is proccesing")
 cv2.waitKey(0)
 cv2.destroyAllWindows()

 temp = calculateResultsFor(image)

 i=i+1
 if temp>max_score :
        max_score = temp
        max_score_id = image


print("the max score is : ",max_score,"|| with the id : ",max_score_id )

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
