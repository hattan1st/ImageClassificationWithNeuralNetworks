import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from Tools.i18n.makelocalealias import optimize
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics
from collections import Counter


(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images/255, testing_images/255

#here we go                                                                                                                                                         ؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟؟ظing to defin a calss name list, and visualize 16 out of all those images from the data set
calss_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Reducing the number of images: here we're reducing the number of images input to the neural network, which may help speed up the training process
training_images = training_images[:60000]
training_labels = training_labels[:60000]
testing_images = testing_images[:15000]
testing_labels = testing_labels[:15000]


#
# #now we build a neural network and for that start by defining as model
# model = models.Sequential()
# model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3) ) ) #2D meaning image with row & col, 32 type of fillters, kernal=3x3, inputs size image= 32x32px with RGB colors
# model.add(layers.MaxPooling2D((2,2))) #kernsl 2x2
# model.add(layers.Conv2D(64, (3,3), activation='relu')) #here no input size cuz the input is layer befor
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax')) #output
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels) )
#
# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")
#
# model.save('image_classifier.keras')


model = models.load_model('image_classifier.keras')

img =cv.imread('ImageTest\cat22.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow (img, cmap=plt.cm.binary)

prediction= model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Prediction is {calss_names[index]}')

plt.show()




#
# # استخراج الأوزان من طبقات النموذج
# weights = model.get_weights()  # قائمة تحتوي على أوزان كل الطبقات
# flattened_weights = np.concatenate([w.flatten() for w in weights if len(w.shape) > 1])  # جمع كل الأوزان في قائمة واحدة
#
# # حساب التكلفة بناءً على الأوزان
# cost = flattened_weights**2  # تكلفة مبنية على مربع الأوزان
#
# # رسم العلاقة بين الأوزان والتكلفة
# plt.figure(figsize=(10, 6))
# plt.plot(flattened_weights, cost, marker='o', linestyle='', color='blue')  # نقاط فقط
# plt.title('Relationship Between Weights and Cost')
# plt.xlabel('Weights')
# plt.ylabel('Cost')
# plt.grid(True)
# plt.show()




