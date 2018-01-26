import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
img=cv.imread('girl.jpg')
#cv.imshow('img',img)
#cv.waitKey(0)
X=img.reshape((img.shape[0]*img.shape[1]),img.shape[2])
K=5;
kmeans=KMeans(n_clusters=K).fit(X)
label=kmeans.predict(X);
img2=np.zeros_like(X)
for k in range(0,K-1):
 img2[label==k]=kmeans.cluster_centers_[k];
 img3=img2.reshape(img.shape[0],img.shape[1],img.shape[2])
cv.imwrite('img3.jpg',img3)
cv.waitKey(0)

 

