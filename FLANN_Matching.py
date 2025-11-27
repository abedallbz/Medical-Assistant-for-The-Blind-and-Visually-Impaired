import pickle
from Image import Image
from glob import glob
import os
import cv2
from Preprocess import process_image
import time
import  numpy as np
# get the dataset
# get the start time
st = time.time()
max_score = 0
sift = cv2.SIFT_create()

#get image from My Medicine folder for testing
query_folder_path = r"C:\Users\abedall\Desktop\MY_VERSIONS\myQuery"
query = cv2.imread(query_folder_path + r"/" + 'DSC_2545.JPG')
query = process_image(query)
keypoint1, descriptor1 = sift.detectAndCompute(query, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=7)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)


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

i = 2332
while i <= 3385 :
 image= i
 def calculateResultsFor(image):
    descriptor2 = fetchDescriptorFromFile(image)
    if np.any( descriptor2 )==  [0] :
        matches =0
        score = 0
        return score
    else :
        matches = calculateMatches(descriptor1, descriptor2)
    #score = calculateScore(len(matches), len(keypoint1), len(keypoint2))
    #plot = getPlotFor(i, j, keypoint1, keypoint2, matches)
    #print(len(matches), len(keypoint1), len(keypoint2), len(descriptor1), len(descriptor2))
    #print(score)
        score = len((matches))
    # print(score)
        return score

 bf = cv2.BFMatcher()


 def calculateMatches(des1, des2):
    matches = flann.knnMatch(des1, des2, k=2)
    topResults = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            topResults.append([m])
    return topResults


# def getPlot(image1, image2, keypoint1, keypoint2, matches):
#     image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#     image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
#     matchPlot = cv2.drawMatchesKnn(image1, keypoint1, image2, keypoint2, matches, None, [255, 255, 255], flags=2)
#     return matchPlot



 cv2.waitKey(0)
 cv2.destroyAllWindows()
 temp = calculateResultsFor(image)

 i = i + 1
 if temp > max_score:
     max_score = temp
     max_score_id = image

print("the max score is : ", max_score, "|| with the id : ", max_score_id)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
