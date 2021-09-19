import cv2
import numpy as np
import glob

file_array = []
img_array = []
for filename in glob.glob('./Diagnostics/local/massratio/25/dist_function/IAT_E2_f2D_*.png'):
    file_array.append(filename)

file_array.sort()
for filename in file_array:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('./output_video_e1_1d.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()