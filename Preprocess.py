import cv2

# Resize images to a similar dimension
# This helps improve accuracy and decreases unnecessarily high number of keypoints


def resize_image(image):
    return cv2.resize(image,(800,600))


def imageResizeTrain(image):
    maxD = 1024
    height, width = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    print(image.shape)
    return image


def imageResizeTest(image):
    maxD = 1024
    height, width, channel = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio), maxD)
    else:
        newSize = (maxD, int(maxD/aspectRatio))
    image = cv2.resize(image, newSize)
    return image


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def process_image(image):
    image_gray = convert_to_gray(image)
    image_resized = resize_image(image_gray)
    return image_resized


def preprocess(image_list):
    imagesBW = []
    for image in image_list:
        imagesBW.append(process_image(image))
    return imagesBW

