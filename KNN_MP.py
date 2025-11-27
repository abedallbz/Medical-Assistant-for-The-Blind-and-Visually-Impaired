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
from multiprocessing import Process



max_score = 0
# max_score_id = -1
# get the start time

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# get the dataset
#query_folder_path = r"C:\Users\co90co\Desktop\graduation project\query"
query_folder_path = r"C:\Users\abedall\Desktop\MY_VERSIONS\myQuery"
query = cv2.imread(query_folder_path + r"/" + 'DSC_2545.JPG')
query = process_image(query)
#cv2.imshow('query', query)

keypoint1, descriptor1 = sift.detectAndCompute(query, None)



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


def calculateResultsFor(queue):
    global max_score
    global max_score_id

    for q in queue:
        #print(q)
        descriptor2 = fetchDescriptorFromFile(q)
        if np.any(descriptor2) == [0]:
            matches = 0
            score = 0
            return score
        else:
            matches = calculateMatches(descriptor1, descriptor2)
            score = len((matches))
            #return score
            if score > max_score:
                max_score = score
                max_score_id =  q



def calculateMatches(des1, des2):
    matches = bf.knnMatch(des1, des2, k=2)
    topResults = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            topResults.append([m])
    return topResults

if __name__ == '__main__':
    max_score = 0
    max_score_id = -1
    i = 2332
    processes = []
    queue1 = []
    queue2 = []
    queue3 = []
    queue4 = []

    for i in range(2332, 2595):
        queue1.append(i)

    for i in range(2595, 2858):
        queue2.append(i)

    for i in range(2858, 3121):
        queue3.append(i)

    for i in range(3121, 3386):
        queue4.append(i)

    st = time.time()

    process1 = Process(target=calculateResultsFor, args = [queue1])
    process2 = Process(target=calculateResultsFor, args = [queue2])
    process3 = Process(target=calculateResultsFor, args = [queue3])
    process4 = Process(target=calculateResultsFor, args = [queue4])

    process1.start()
    process4.start()
    process3.start()
    process2.start()

    process1.join()
    process4.join()
    process3.join()
    process2.join()


    # for process in processes:
    #     process.join()

    print("the max score is : ", max_score, "|| with the id : ", max_score_id )

    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')



    cv2.waitKey(0)
    cv2.destroyAllWindows()