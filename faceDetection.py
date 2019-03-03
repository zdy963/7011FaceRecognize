# import required packages
import os
import cv2
import re
import dlib
from imutils import paths
import argparse
import time
from tensorflow.python.keras.preprocessing.image import img_to_array
from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool

# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

# initialize hog + svm based face detector
# hog_face_detector = dlib.get_frontal_face_detector()


def expandLanger(cor, percent=(0.1,0.15)):
    """
    Expand the image specific percent larger
    :param cor: face coordinate
    :param percent: expand percentage
    :return: coordinate
    """
    w = cor[2]
    h = cor[3]

    x = cor[0] - int(percent[0]*w)
    y = cor[1] - int(percent[1]*h)
    w = w + int(percent[0]*w)*2
    h = h + int(percent[1]*h)*2

    return [x, y, w, h]


def faceDetect(imagePath):
    """
    Detect face in image
    :param image:
    :param IMAGE_DIMS:
    :return: image array
    """
    cropPath = imagePath.split('.')
    if not cropPath[0].endswith('cropped'):
        croppedPath = cropPath[0] + '_cropped.' +cropPath[1]
        if not os.path.exists(croppedPath):
            image = cv2.imread(imagePath)  # Load the image
            # print(cropPath[0].split('/')[-1],end='')
            # Apply face detection
            # Upsample the image 1 time
            # cnn = True

            height,width,depth = image.shape
            image2 = cv2.pyrDown(image)
            if height*width > 8e6:
                print('Too large', end=' ')
                image2 = cv2.pyrDown(image2)
                faces = cnn_face_detector(image2, 1)

                # faces = hog_face_detector(image, 0)
                face_dict = {'lst': []}
                for face in faces:
                    x = face.rect.left()
                    y = face.rect.top()
                    w = face.rect.right() - x
                    h = face.rect.bottom() - y
                    face_dict['lst'].append(w * h)
                    face_dict[w * h] = {'cor': [x*4, y*4, w*4, h*4]}

            else:
                faces = cnn_face_detector(image2, 1)

                # faces = hog_face_detector(image, 0)
                face_dict = {'lst':[]}
                for face in faces:
                    x = face.rect.left()
                    y = face.rect.top()
                    w = face.rect.right() - x
                    h = face.rect.bottom() - y
                    face_dict['lst'].append(w*h)
                    face_dict[w*h] = {'cor':[x*2, y*2, w*2, h*2]}

            try:
                cor = face_dict[max(face_dict['lst'])]['cor']
                # cor = expandLanger(cor, percent=(0.08, 0.12))
                x = cor[0]
                y = cor[1]
                w = cor[2]
                h = cor[3]
                image = image[y:y + h, x:x + w]
                imagePath = imagePath.replace('photos2/', 'cropped_photos2_origin/')
                # print(imagePath)
                imagePath = imagePath.split('.')
                cv2.imwrite(imagePath[0] + '_cropped.' + imagePath[1], image)
            except ValueError:
                print(imagePath, 'No output',end=' ')
            # print('\r')

            # image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))  # Resize it
            # image = img_to_array(image)  # Transform into array


def delete_photo():
    print('Deleting Cropped Picture')
    imagePaths = sorted(list(paths.list_images('photos')))
    for imagePath in imagePaths:
        cropPath = imagePath.split('.')
        if cropPath[0].endswith('cropped'):
            os.remove(imagePath)

    print('Deleting Origin Picture')
    imagePaths = sorted(list(paths.list_images('photos_cropped')))
    for imagePath in imagePaths:
        cropPath = imagePath.split('.')
        if not cropPath[0].endswith('cropped'):
            os.remove(imagePath)


if __name__ == '__main__':
    # pool = Pool(2)
    imagePaths = sorted(list(paths.list_images('photos2')))

    imageCount = len(imagePaths)
    current = 0
    for imagePath in imagePaths:
        current += 1
        percent = current / imageCount
        print('Finished %' + str(round(percent*100, 1)), imagePath, end=' ')
        start = time.time()
        faceDetect(imagePath)
        end = time.time()
        print(format(end - start, '.2f'),'s')

    # delete_photo()