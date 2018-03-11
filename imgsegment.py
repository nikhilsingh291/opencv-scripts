import sklearn.cluster 
import numpy as np
import cv2
import matplotlib.pyplot as plt


print "input image name:"
img=raw_input()
print "input number of clusters:"
k=int(raw_input())


dat=cv2.imread(img)

cv2.imshow("original image",dat)
cv2.waitKey(0)
cv2.destroyAllWindows()

w,h,d= tuple(dat.shape)
print w,h,d
print "Shape of image"

#original_shape = dat.shape # so we can reshape the labels later
im=np.reshape(dat, (w * h, d))

print "flatten",im.shape

#declaring k-means classifier 
clf = sklearn.cluster.KMeans(n_clusters=k)
labels= clf.fit_predict(im)


def partimage(imo,n):
    pi=imo
    image = np.zeros(((w*h), d))
    p=0
    for lv in labels:
        if(lv==n):
            image[p]=pi[p]
        else:
            pass
        p=p+1
    return image

 
def recreate_image(color, labels, w, h):
##    """Recreate the (compressed) image from the cluster center & labels"""
    print color
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = color[labels[label_idx]]
            label_idx += 1
    return image


plt.imshow(recreate_image(clf.cluster_centers_, labels, w, h))
plt.show()


for z in range(0,k): 
    newimg=partimage(im,z).reshape(w,h,3)
    plt.imshow(newimg)
    plt.show()
