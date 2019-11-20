from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os.path
import matplotlib.pyplot as plt

labels = pd.read_csv("D:\\grozi\\index2\\UPC_index.txt",sep="\t")
print(labels.shape)
print(labels.head(3))

tag=[]
train_image = []
for i in (range(1,121)):
    j=1
    while(j!=0):
        file_path='D:\\grozi\\inSitu\\'+str(i)+'\\video\\video'+str(j)+'.png'
        if os.path.isfile(file_path):
            img = image.load_img(file_path, target_size=(28,28,1), grayscale=False)
            img = image.img_to_array(img)
            img = img/255
            train_image.append(img)
            tag.append(i-1)
            j=j+1
        else:
            j=0

print("tag length:",len(tag))
print("Total Images: ",len(train_image))
list_of_tuples=list(zip(train_image,tag))
df = pd.DataFrame(list_of_tuples, columns = ['img', 'label'])
print(df.shape)

X = np.array(train_image)
y=pd.get_dummies(df['label']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

print("Train Images:",X_train.shape[0])
print("Test Images:",X_test.shape[0])

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=(28,28,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(120, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

history=model.fit(X_train, y_train, epochs=5)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:"+"%s: %.2f%%" % (model.metrics_names[1], (scores[1])*100))

y_pred=model.predict_classes(X_test)
for i in (range(0,5)):
    j=y_pred[i]
    classname=labels['product_name'][j]
    print("Predicted class for image {} are: {}".format(i,classname))
   

import cv2
tag1=[]
test_image = []
nemos_friends = []
for i in (range(1,11)):
    plt.subplot(1, 2, 1)
    file_path='D:\\grozi\\New Folder\\p'+str(i)+'.png'
    img = image.load_img(file_path, target_size=(28,28,1), grayscale=False)
    img1 = image.img_to_array(img)
    img1 = img1/255
    test_image.append(img1)
    friend = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    nemos_friends.append(friend)
    plt.subplot(1, 2, 1)
    plt.imshow(nemos_friends[i-1])
    plt.show()
test = np.array(test_image)
y_pred=model.predict_classes(test)
for i in (range(0,10)):
    j=y_pred[i]
    print("Predicted class for image {} are: {}".format(i,labels['product_name'][j]))


