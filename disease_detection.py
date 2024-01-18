import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

training_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = training_data_generator.flow_from_directory(
    'data/train',
    target_size=(64,64),
    batch_size=32,
    class_mode='sparse'
)

test_data_generator = ImageDataGenerator(
    rescale=1./255
)

test_set = test_data_generator.flow_from_directory(
    'data/test',
    target_size=(64,64),
    batch_size=32,
    class_mode='sparse'
)

cnn = tf.keras.models.Sequential()                                                                     #type: ignore
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", input_shape=[64, 64, 3])) #type: ignore
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))                                             #type: ignore
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))                          #type: ignore
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))                                             #type: ignore
cnn.add(tf.keras.layers.Flatten())                                                                     #type: ignore
cnn.add(tf.keras.layers.Dense(units=32, activation="relu"))                                            #type: ignore
cnn.add(tf.keras.layers.Dense(units=5, activation="softmax"))                                          #type: ignore
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn.fit(x=training_set, validation_data=test_set, epochs=25)
cnn.save('model.h5')