import matplotlib.pyplot as plt , matplotlib.image as iimg
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

#loading the data using only 6000 images
all_images=pd.read_csv('train.csv')
images=all_images.iloc[0:6000,1:]
label=all_images.iloc[0:6000,:1]
train_images , test_images , train_label , test_label=train_test_split( images,label,train_size=0.8,random_state=0 )

#Viewing an Image
n=1
img=train_images.iloc[n].to_numpy()
img=img.reshape((28,28))

"""
plt.imshow(img,cmap='gray')
plt.title(train_label.iloc[n,0])
"""




#Training the model
clf=svm.SVC()
clf.fit(train_images,train_label.values.ravel())

"""
score is 0.951.
We're trying another way to raise the score.
we'll make the images white and black,any pixel with a value simply becomes 1 and everything else remains 0. 
"""






test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[n].to_numpy()
img=img.reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_label.iloc[n,0])




#With playing with the parameters of svm.
clf=svm.SVC(kernel='rbf',C=7,gamma=0.009)
clf.fit(train_images,train_label.values.ravel())
#clf.score(test_images,test_label)
print("score is: ", clf.score(test_images,test_label))
#score becomes 0.955


#labeling data
test_data=pd.read_csv('test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])
print(results)

df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
print(df.head(10))





