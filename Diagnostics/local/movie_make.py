import cv2
import numpy as np
import glob

file_array = []
img_array = []
for filename in glob.glob('./Diagnostics/local/Cori/mass25/rescheck/4/dist_function/M25_E2_3_f2D_*.png'):
    file_array.append(filename)

file_array.sort()
for filename in file_array[:]:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
out = cv2.VideoWriter('./output_video_e2_1d.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for image in img_array:
    out.write(image)
cv2.destroyAllWindows()
out.release()