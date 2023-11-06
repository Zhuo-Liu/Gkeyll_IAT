import cv2
import numpy as np
import glob

file_array = []
img_array = []
# for filename in glob.glob('./python_file_unitility/Cori/mass25/rescheck/4/dist_function/M25_E2_3_elcps_*.png'):
#     file_array.append(filename)

for filename in glob.glob('./python_file_unitility/Cori/mass25/rescheck/4/dist_function/M25_E2_3_f2D_*.png'):
    file_array.append(filename)

file_array.sort()
img = cv2.imread(file_array[0])
height, width, layers = img.shape
#size = (width,height)
#size = (806,height)
size = (1200, height)

for filename in file_array[:]:
    img = cv2.imread(filename)
    # img = cv2.resize(img,size)
    img = img[:,:1200,:]
    #mg = cv2.resize(img,size)
    img_array.append(img)
 
out = cv2.VideoWriter('./output_video_m25e5.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
 
for image in img_array:
    out.write(image)
cv2.destroyAllWindows()
out.release()