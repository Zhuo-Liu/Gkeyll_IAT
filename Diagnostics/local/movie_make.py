import cv2
import numpy as np
import glob
 
img_array = []
for filename in glob.glob('./Diagnostics/local/dist_function/IAT_E2_f1D_*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('./output_video_e2_1d.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()