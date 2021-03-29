import numpy as np
import cv2
from sklearn.utils import shuffle
from scipy.ndimage import distance_transform_edt
from skimage.transform import resize
import os
import time
from tensorflow import ConfigProto, InteractiveSession
from skimage import img_as_float
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras import models

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

########################################################################################################################
##################################################--GLOBAL PARAMETERS--#################################################
########################################################################################################################

dir_test_imgs_name = 'Test_Images'
dir_test_segs_name = 'Test_Ground_Truth'

input_height, input_width = 224, 224
IMAGE_ORDERING = 'channels_last'

decay = 7
count = 0


########################################################################################################################
#############################################--IMAGE_PREPERATION--######################################################
########################################################################################################################

def get_dist_maps(coords, mask,  shp=(224, 224)):


    distance = mask
    print(f"Coords length: {len(coords)}")
    distance = cv2.bitwise_not(distance)

    distance = distance_transform_edt(distance)
    print(f"Cords after distance transform data type : {type(distance)}")
    print(f"Cords after distance transform shape : {np.shape(distance)} {distance.ndim}")

    distance = distance[:, :, np.newaxis]


    if shp != (224, 224):
        distance = resize(1 - distance / np.max(distance), (224, 224, 1)) ** decay

    else:
        distance = (1 - distance / np.max(distance)) ** decay


    print(f"Cords before return distance transform data type : {type(distance)}")
    print(f"Cords before return distance transform shape : {np.shape(distance)} {distance.ndim}")
    return distance

def read_test_image_dataset(rgb_dir_name, ground_truth_target_dir, width, height, verbose=False):

    # Declare lists for data input
    X = []
    X_filepaths = []
    Y = []
    Y_filepaths = []
    T = []
    T_filepaths = []
    true_distance_maps = []
    C = []

    rgb_dir_path = os.path.join(os.getcwd() + str('/') + rgb_dir_name + str('/'))
    ground_truth_target_dir_path = os.path.join(os.getcwd() + str('/') + ground_truth_target_dir + str('/'))

    # Create lists of image files in directories
    rgb_filenames_list = os.listdir(rgb_dir_path)
    ground_truth_target_filenames_list = os.listdir(ground_truth_target_dir_path)

    # Short lists of image files in directories
    rgb_filenames_list.sort()
    ground_truth_target_filenames_list.sort()

    print(f"Test photos: {rgb_filenames_list}")
    print(f"Test photos: {ground_truth_target_filenames_list}")

    # Loop in each file pair in two directories
    for img, target in zip(rgb_filenames_list, ground_truth_target_filenames_list):

        # Create full paths for each pair
        img_file_path = os.path.join(rgb_dir_path + img)
        annot_file_path = os.path.join(ground_truth_target_dir_path + target)

        X_filepaths.append(img_file_path)
        Y_filepaths.append(annot_file_path)

        # Read RGB image and resize
        rgb_img = cv2.imread(img_file_path, 1)
        rgb_img = cv2.resize(rgb_img, (width, height))

        rgb_img = img_as_float(rgb_img)

        # Read Anotation image and resize
        annot_img = cv2.imread(annot_file_path, 1)
        annot_img = cv2.resize(annot_img, (width, height))
        annot_img_rgb = annot_img.copy()
        annot_img = cv2.cvtColor(annot_img, cv2.COLOR_BGR2GRAY)

        annot_img_copy = annot_img.copy()

        # Threshold ground truth image and apply morphological filters for stem segmentation
        ret, thresh1 = cv2.threshold(annot_img, 10, 255, cv2.THRESH_BINARY)

        kernel = np.ones((1, 1), np.uint8)
        thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        thresh1 = cv2.dilate(thresh1, kernel, iterations=1)

        print(f"File_Name: {target}")

        # Find stem contour and draw it to black mask
        contours = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        bud_area_mask = np.ones(thresh1.shape, thresh1.dtype)
        cv2.drawContours(bud_area_mask, contours, -1, 255, -1)

        print(f"LEN OF THE CONTOURS {len(contours)}")


        # Compute distance map of stem contour
        distance = get_dist_maps(coords=contours, mask=thresh1)
        distance = cv2.convertScaleAbs(distance, alpha=(255.0))

        if verbose:
            cv2.imshow("Annotation Image Threshold", thresh1)
            cv2.imshow("Annotation Image", annot_img_copy)
            cv2.imshow("Bud Area Mask", bud_area_mask)
            cv2.imshow("distance map", distance)
            cv2.imshow("rgb", rgb_img)
            cv2.imwrite("ground_truth_distance_map.png", annot_img_rgb)
            cv2.imwrite("rgb_image.png", rgb_img)
            cv2.imwrite("distance_map.png", distance)
            cv2.waitKey(0)

        print(f"Function distance transform type : {type(distance)}")
        print(f"Function distance transform shape : {np.shape(distance)}")

        # Append image data in lists
        X.append(rgb_img)
        Y.append(distance)
        T.append(annot_img)
        C.append(contours)
        true_distance_maps.append(distance)


    X2, Y2 = np.array(X), np.array(Y)

    print(f" Test Image Dataset shape for train {X2.shape} --- {Y2.shape}")

    return X2, Y2, X_filepaths, Y, T_filepaths, true_distance_maps, T, C

########################################################################################################################
####################################################--METRICS--#########################################################
########################################################################################################################
def R_squared(y, y_pred):
  residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
  total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
  r2 = tf.subtract(1.0, tf.truediv(residual, total))
  return r2


########################################################################################################################
##############################################--MODEL_EVALUATION--######################################################
########################################################################################################################
# Read test image pairs from directory and if verbose is desired turn parameter to True
X_test, Y_test, X_filepath, Y_filepath, True_filepath, true_distance_map, Seg_img, Seg_C = read_test_image_dataset(rgb_dir_name=dir_test_imgs_name,
                                                                                ground_truth_target_dir=dir_test_segs_name,
                                                                                width=input_width, height=input_height, verbose=False)

# Load model from file
vine_model = models.load_model('UNET_mobilenetv2.h5', custom_objects={'R_squared': R_squared})
vine_model.summary()
pred = vine_model.predict(X_test)

# Loop in each image pair
for img_target, img_true, img_true_path, sg_gt_img in zip(X_test, Y_test, true_distance_map, Seg_img):
    count += 1

    # Predict and calculate time
    t1 = time.time()
    dist_map_pred = vine_model.predict(img_target[np.newaxis,:,:,:])
    pred_map = dist_map_pred[0, :, :, 0]
    t2 = time.time()
    pred_time = round((t2 - t1), 2)
    print(f"Pred_time: {pred_time}")

    # Show predictions and ground truth images
    cv2.imshow("Predicted", pred_map)
    cv2.imshow("Ground_Truth_Distance_Map", img_true_path)
    cv2.imshow("Ground_Truth", sg_gt_img)
    cv2.waitKey(0)









