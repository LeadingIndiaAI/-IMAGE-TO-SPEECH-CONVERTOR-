import glob
import cv2
import numpy as np

# for i in range(1,10):
# 	path = 'Fnt/Sample00'+str(i)+'/*.png'
# for i in range(10,63):
# 	path = 'Fnt/Sample0'+str(i)+'/*.png'

# images = glob.glob(path)
# #print(images)
# #print(len(images))
# for img in images:
# 	#print(img)
# 	file = cv2.imread(img, 0)
# 	file = np.resize(file, (64, 64, 1))
# 	cv2.imwrite('newdata/Sample 1/img.png',file)
# 	print(img,file.shape)

for i in range(1,10):
	path = 'Fnt/Sample00'+str(i)+'/*.png'
	images=glob.glob(path)
	ctr=1
	for img in images:
		ctr=ctr+1
		file = cv2.imread(img, 0)
		file = np.resize(file, (64, 64, 1))
		cv2.imwrite('newdata/Sample '+str(i)+'/img'+str(ctr)+'.png',file)
		print(img,ctr,file.shape)


