import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("/home/noldsoul/Desktop/model_300k.h5", custom_objects={"tf": tf})
model.summary()

# images_path = "/home/vineeth_s_subramanyan/Downloads/Datasets/Mapillary/training/images/"
# labels_path = "/home/vineeth_s_subramanyan/Downloads/Datasets/Mapillary/training/instances/"

# for file in glob.glob(images_path + "*.jpg"):
#     print(file)
#     img = cv2.resize(cv2.imread(file),(640,480))
#     label_file = file.split("/")[-1].split(".")[0] + ".png"
#     frame = expand_dims(img,axis=0)
#     label = cv2.resize(cv2.imread(labels_path + label_file),(640,480))
#     cv2.imshow("Image",img)	
#     out = model.predict(frame)[0]
#     ret,out = cv2.threshold(out,1,255,cv2.THRESH_BINARY)   
#     label_list = [[2,24,9],[13]]
#     kernel1 = np.ones((1,1),np.uint8)
#     kernel2 = np.ones((3,3),np.uint8)
#     mask = []
#     background_mask = np.zeros_like(label)
#     for label_id in label_list:
#         color_array = np.zeros_like(label)
#         for idx in label_id:
#             color_array[label == idx] = 255
#             color_array = color_array[:,:,0]
#             color_array = cv2.erode(color_array,kernel1,iterations = 6)
#             color_array = np.dstack((color_array,color_array,color_array))
#         background_mask = cv2.bitwise_or(background_mask, color_array)
#         mask.append(color_array[:,:,0])
        

#     background_mask = cv2.bitwise_not(background_mask[:,:,0])
#     background_mask = cv2.dilate(background_mask,kernel2,iterations = 1)
#     background_mask = cv2.erode(background_mask,kernel2,iterations = 2)

#     final_mask = np.zeros((img.shape[0],img.shape[1],len(mask)+1))
#     final_mask[:,:,0] = background_mask

#     for i in range(len(mask)):
#         final_mask[:,:,i+1] = mask[i]

#     for j in range(final_mask.shape[2]):
#         current_mask = final_mask[:,:,j]
#         test_img = current_mask.copy()
#         test_img[:400,:] = 0
#         cv2.imshow("Ground Truth",current_mask)
#         cv2.imshow("Prediction",test_img)
#         IoU,precision,recall,F1 = get_iou(current_mask,test_img)
#         print("Label"+str(j),IoU,precision,recall,F1)
#         k = cv2.waitKey()
#         if(k==ord('q')):
#             exit()
#         elif(k==ord('p')):
#             cv2.waitKey()