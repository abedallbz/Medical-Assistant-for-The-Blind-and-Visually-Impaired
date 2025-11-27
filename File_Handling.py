import os
import cv2
import pickle
from Image import Image
from glob import glob
import os
import cv2
from Preprocess import process_image
import time

all_images_list =[]
rootdir = 'C:/Users/abedall/Desktop/graduiton project/dataset/هضمية'
for file in os.listdir(rootdir):
 d = os.path.join(rootdir, file)
 folder_path = d


 def get_images():

    images_list = []
    for image_name in os.listdir(folder_path):
        if image_name.endswith(".png") or image_name.endswith(".JPG") or image_name.endswith(".jpeg"):
            image_path = folder_path + r"/" + image_name
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                cur_image = Image(image)
                cur_image.set_name(image_name)
                images_list.append(cur_image)

            else:
                print('image path not exist ', image_path)
    return images_list

  # get the dataset
 images_list = get_images()
 all_images_list.append(images_list)


print(len(all_images_list))
def save_keypoints(all_images_list):
   for images_list in all_images_list :
    for image in images_list:
        keypoints_infos = []
        filepath = r"C:/Users/abedall/Desktop/graduiton project/dataset/features_extraxtion/keypoints/" + str(image.name.split('.')[0]) + "_kps" + ".json"
        for point in image.keypoints:
            temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
            keypoints_infos.append(temp)
        with open(filepath, 'ab') as fp:
            pickle.dump(keypoints_infos, fp)
        fp.close()


def save_descriptors(all_images_list):
  for images_list in all_images_list:
    for image in images_list:
        filepath = r"C:/Users/abedall/Desktop/graduiton project/dataset/features_extraxtion/descriptors/" + str(image.name.split('.')[0]) + "_des" + ".json"
        with open(filepath, 'ab') as fp:
            pickle.dump(image.descriptors, fp)
        fp.close()


def fetchKeypointFromFile(imagename):
    filepath = r"C:/Users/abedall/Desktop/graduiton project/dataset/features_extraxtion/keypoints/" + "DSC_" + str(imagename)+ "_kps" + ".json"
    keypoint = []
    file = open(filepath, 'rb')
    deserializedKeypoints = pickle.load(file)
    file.close()
    for point in deserializedKeypoints:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        keypoint.append(temp)
    return keypoint


def fetchDescriptorFromFile(imagename):

    filepath = r"C:/Users/abedall/Desktop/graduiton project/dataset/features_extraxtion/descriptors/" + "DSC_" + str(imagename) + "_des" + ".json"
    file = open(filepath, 'rb')
    descriptor = pickle.load(file)
    file.close()
    return descriptor

sift = cv2.SIFT_create()


def compute_sift(img):
    return sift.detectAndCompute(img, None)


# get the dataset
# images_list = get_images()

#pre-processing
for images_list in all_images_list :
 for image in images_list:
    image.set_process_image(process_image(image.value))


# extracting the features for each image
for images_list in all_images_list :
 for image in images_list:
    kps, des = compute_sift(image.process_image)
    image.set_keypoints(kps)
    image.set_descriptors(des)


# saving the keypoints and descriptors for each image in  a file
save_descriptors(all_images_list)
save_keypoints(all_images_list)

