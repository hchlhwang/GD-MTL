import sys
import numpy as np
from os import listdir
from os.path import isfile, join
np.set_printoptions(threshold=sys.maxsize)


# Set data path
depthPath = '/home/hochul/Repository/mtan/im2im_pred/preprocessed/train/depth/'
imgPath = '/home/hochul/Repository/mtan/im2im_pred/preprocessed/train/image/'
labelPath = '/home/hochul/Repository/mtan/im2im_pred/preprocessed/train/label/'
normalPath = '/home/hochul/Repository/mtan/im2im_pred/preprocessed/train/normal/'

# Modality list
depthFileList = [f for f in listdir(depthPath) if isfile(join(depthPath, f))]
imageFileList = [f for f in listdir(imgPath) if isfile(join(imgPath, f))]
labelFileList = [f for f in listdir(labelPath) if isfile(join(labelPath, f))]
normalFileList = [f for f in listdir(normalPath) if isfile(join(normalPath, f))]

# Modality file
for file in depthFileList:
  depthData = np.load(depthPath+file)

for file in imageFileList:
  imageData = np.load(imgPath+file)

for file in labelFileList:
  labelData = np.load(labelPath+file)

for file in normalFileList:
  normalData = np.load(normalPath+file)

# Print data info
print("==== Depth ====") # 1.5 ~ 2.1 ?
print(f"Data num: {len(depthFileList)}\nData shape: {depthData.shape}\n")

print("==== Image ====") # 0 ~ 1
print(f"Data num: {len(imageFileList)}\nData shape: {imageData.shape}\n")

print("==== Label ====") # -1 ~ 12
print(f"Data num: {len(labelFileList)}\nData shape: {labelData.shape}\n")

print("==== Normal ====") # -1 ~ 1
print(f"Data num: {len(normalFileList)}\nData shape: {normalData.shape}\n")
