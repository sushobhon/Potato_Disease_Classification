# importing necessary libraries
from tensorflow.keras import models, layers, Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# Defining some constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 30
CHANNELS = 3

# Importing Data into tensorflow data object
dataset = tf.keras.preprocessing.image_dataset_from_directory(directory= "./PlantVillage__Potato",
                                                              seed= 123,
                                                              shuffle= True,
                                                              batch_size= BATCH_SIZE,
                                                              image_size= (IMAGE_SIZE, IMAGE_SIZE))

class_names = dataset.class_names

# Length of the data set is 68 because the data is loaded of batch size 32. So 32*68 = 2176
len(dataset)

for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy())

# Visulizing Some of the images
plt.figure(figsize=(10,10))
for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(4, 3, i+1)
        plt.imshow(image_batch[i].numpy().astype('uint8'))
        plt.title(class_names[label_batch[i].numpy()])
        plt.axis("off")
plt.show()

# Spliting the dataset into 80% training, 10% validation and 10% test.
# train_ds = dataset.take(54)
# len(train_ds)

def get_train_test_split(dataset, train_size = 0.8, validation_size = 0.1, test_size = 0.1, suffle = True, suffle_size = 1000):

    assert train_size + test_size + validation_size == 1

    if suffle:
        dataset = dataset.shuffle(suffle_size, 12)

    train_size = int(len(dataset) * train_size)
    val_size = int(len(dataset) * validation_size)

    print(f"train size {train_size}")
    print(f"Validation size {val_size}")

    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_train_test_split(dataset= dataset)

len(train_ds)
len(val_ds)
len(test_ds)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

## Model Building
# Creating a layers or resizing and normalizing the data

resizing_and_normalizing = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1./255)
])

# Data Augmentation
data_augumentation = tf.keras.Sequential([
    layers.RandomFlip(mode="horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])

# Applying Data Augumentation on train dataset
train_ds = train_ds.map(
    lambda x, y: (data_augumentation(x, training = True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

# specifying model architecture

n_class = len(class_names)
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

model = Sequential([
    resizing_and_normalizing,
    tf.keras.layers.Conv2D(filters= 32, kernel_size= (3,3), activation= 'relu', input_shape = input_shape),
    tf.keras.layers.MaxPool2D(pool_size= (2,2)),
    tf.keras.layers.Conv2D(filters= 64, kernel_size= (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(pool_size= (2,2)),
    tf.keras.layers.Conv2D(filters= 64, kernel_size= (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(pool_size= (2,2)),
    tf.keras.layers.Conv2D(filters= 64, kernel_size= (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(pool_size= (2,2)),
    tf.keras.layers.Conv2D(filters= 64, kernel_size= (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(pool_size= (2,2)),
    tf.keras.layers.Conv2D(filters= 64, kernel_size= (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(pool_size= (2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation= 'softmax')
])

model.build(input_shape = input_shape)

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(train_ds,
                    batch_size = BATCH_SIZE,
                    epochs = EPOCHS,
                    validation_data = val_ds,
                    verbose = 1
                    )

# Performace in test set
model.evaluate(test_ds)

# # Run Prediction on Sample image
# for image_batch, label_batch in test_ds.take(1):
#     first_image = image_batch[5].numpy().astype('uint8')
#     first_label = label_batch[5].numpy()

#     plt.imshow(first_image)
#     pred_label = np.argmax(model.predict(image_batch)[5])
#     plt.title(f'Actual: {class_names[first_label]}\nPredicted: {class_names[pred_label]}')
#     plt.axis("off")

# plt.show()

# Writting a function to predict and show images
def predict(model, img, class_names):
    img_array = tf.keras.preprocessing.image.img_to_array(img= img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predict_class = class_names[np.argmax(predictions)]
    confidence = round(predictions[0][np.argmax(predictions[0])]*100, 2)

    return predict_class, confidence


plt.figure(figsize= (20,15))
for image_batch, label_batch in test_ds.take(1):
    for i in range(8):
        ax = plt.subplot(2,4,i+1)
        plt.imshow(image_batch[i].numpy().astype('uint8'))
        pred_class, conf = predict(model, image_batch[i].numpy(), class_names)
        actual_class = class_names[label_batch[i]]
        plt.title(f'Actual: {actual_class}\nPredicted: {pred_class}\nConfidence: {conf}%')
        plt.axis('off')

plt.show()

# Saving model
model_version = max([i for i in os.listdir('./Models')])+1
model.save(f"./Models/{model_version}")

model.save("./potatoes.h5")