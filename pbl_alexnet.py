from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os.path
import matplotlib.pyplot as plt
import keras
labels = pd.read_csv("D:\\index2\\UPC_index.txt",sep="\t")
print(labels.shape)
print(labels.head(3))

tag=[]
train_image = []
for i in (range(1,121)):
    j=1
    while(j!=0):
        file_path='D:\\inSitu\\'+str(i)+'\\video\\video'+str(j)+'.png'
        if os.path.isfile(file_path):
            img = image.load_img(file_path, target_size=(3,64,64), grayscale=False)
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

#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(3,64,64), kernel_size=(11,11),activation='relu',strides=(4,4), padding='valid'))

# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11),activation='relu', strides=(1,1), padding='valid'))
model.add()
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3),activation='relu', strides=(1,1), padding='valid'))
model.add(activation='relu')

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3),activation='relu', strides=(1,1), padding='valid'))
model.add(activation='relu')

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3),activation='relu', strides=(1,1), padding='valid'))
model.add(activation='relu')
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(400, input_shape=(64*64*3,)))
model.add(activation='relu')
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(400))
model.add(activation='relu')
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
'''model.add(Dense(1000))
model.add(activation='relu')
# Add Dropout
model.add(Dropout(0.4))'''

# Output Layer
model.add(Dense(120))
model.add(activation='softmax')

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train, y_train, epochs=5)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:"+"%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

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
    file_path='D:\\New Folder\\p'+str(i)+'.png'
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
