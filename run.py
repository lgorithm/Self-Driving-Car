# pip3 install opencv-python

import tensorflow as tf
import scipy.misc
import cv2
import math
import imageio
import numpy as np

model = tf.keras.models.load_model("save/model_final.h5")

img = cv2.imread('steering_wheel.jpg', 0)
rows, cols = img.shape

smoothed_angle = 0

# read data.txt
xs = []
ys = []
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        ys.append(float(line.split()[1]) * math.pi / 180)

# get number of images
num_images = len(xs)

i = math.ceil(num_images * 0.8)
print("Starting frame of video:", str(i))

while(cv2.waitKey(10) != ord('q')):
    full_image = imageio.imread(xs[i])
    image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
    image = np.expand_dims(image, axis=0)
    
    degrees = model.predict(image)[0][0] * 180.0 / math.pi
    print("Steering angle:", str(degrees), "(pred)\t", str(ys[i] * 180 / math.pi), "(actual)")
    
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -smoothed_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("steering wheel", dst)
    
    i += 1

cv2.destroyAllWindows()

