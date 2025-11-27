
import cv2
from Preprocess import process_image
from File_Handling import get_images, save_keypoints, save_descriptors
import time


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

