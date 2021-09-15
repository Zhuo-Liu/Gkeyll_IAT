import cv2
import numpy as np
import glob
import os

img_array = []
for filename in glob.glob('./Diagnostics/local/E2_nu0.0001_Cori/high/dist_function/IAT_E2_f2D_*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('./output_video_e2_nu0.0001_2d.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for image in img_array:
    out.write(image)
cv2.destroyAllWindows()
out.release()