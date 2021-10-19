import cv2
import numpy as np
import glob

file_array = []
img_array = []
for filename in glob.glob('./Diagnostics/local/Ratio/mass400/dist_function/IAT_E2_f2D_*.png'):
    file_array.append(filename)

file_array.sort()
for filename in file_array[:]:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('./output_video_e2_2d2.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for image in img_array:
    out.write(image)
cv2.destroyAllWindows()
out.release()