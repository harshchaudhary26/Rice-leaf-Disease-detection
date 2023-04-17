import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
[3]
IMAGE_SIZE=256
BATCH_SIZE=32  # numher of samples
CHANNELS=3
EPOCHS=100
[4]
dataset= tf.keras.preprocessing.image_dataset_from_directory(
"D:/M2_Test_RC",
   shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
Found 1165 files belonging to 2 classes.
[5]
class_names=dataset.class_names
class_names
['Healthy', 'Hispa']
[6]
len(dataset)
37
[7]
for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy())
(32, 256, 256, 3)
[1 1 1 1 0 0 0 0 1 1 0 0 0 1 1 0 0 1 1 0 0 1 1 1 1 1 0 0 0 0 0 0]
[8]
for image_batch, label_batch in dataset.take(1):
    print(image_batch[0])
tf.Tensor(
[[[255.      255.      255.     ]
  [255.      255.      253.8125 ]
  [255.      255.      255.     ]


[9]
for image_batch, label_batch in dataset.take(1):
    print(image_batch[0].numpy)
<bound method _EagerTensorBase.numpy of <tf.Tensor: shape=(256, 256, 3), dtype=float32, numpy=
array([[[ 57.033203,  75.0332  ,  14.974609],
        [ 51.63086 ,  69.63086 ,   9.392578],
        [ 60.36035 ,  77.54297 ,  21.672852],
        ...,

        [202.06934 , 204.7666  , 209.7666  ],
        [208.52246 , 209.5     , 214.59082 ],
        [210.03125 , 213.03125 , 218.0918  ]]], dtype=float32)>>
[10]
for image_batch, label_batch in dataset.take(1):
    print(image_batch[0].shape)
(256, 256, 3)
[11]
train_size=0.8
len(dataset) * train_size
29.6
[12]
train_ds=dataset.take(54)
len(train_ds)
37
[13]
test_ds=dataset.skip(54)
len(test_ds)
0
[14]
val_size=0.1
len(dataset)*val_size
3.7
[15]
val_ds=test_ds.take(6)
len(val_ds)
0
[16]
test_ds=test_ds.skip(6)
len(test_ds)
0
[17]
def get_dataset_partitions_tf(ds,train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    
    ds_size=len(ds)
    
    if shuffle:
        ds=ds.shuffle(shuffle_size, seed=12)
        
    train_size=int(train_split*ds_size)
    val_size=int(val_split*ds_size)
    
    train_ds  = ds.take(train_size)
    val_ds    = ds.skip(train_size).take(val_size)
    test_ds    = ds.skip(train_size).skip(val_size)
    
    
    
    return train_ds, val_ds, test_ds
[18]
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
[19]
len(train_ds)
29
[20]
len(val_ds)
3
[21]
len(test_ds)
5
[22]
train_ds= train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds= val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds= test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
[23]
resize_and_rescale = tf.keras.Sequential([
    
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),  
    layers.experimental.preprocessing.Rescaling(1.0/255)   
    
])
[24]
data_augmentation =  tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
     
    
])
[25]
input_shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model=models.Sequential([
    resize_and_rescale,
    data_augmentation,
    # l1
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape=input_shape),
    layers.MaxPooling2D((2,2)),  # --> downsample (reduces) the vectors
    # l2
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    # l3
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    # l4
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    # l5
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    # l6
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),  # mulitdimention val to single dimention

    # Dense 1
    layers.Dense(64, activation='relu'),
    # Dense 2
    layers.Dense(n_classes, activation='softmax'), # vetors to probabilities 
    
    
])

model.build(input_shape=input_shape)
[26]
model.summary()
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 sequential (Sequential)     (32, 256, 256, 3)         0         
                                                                 
 sequential_1 (Sequential)   (32, 256, 256, 3)         0         
                                                                 
 conv2d (Conv2D)             (32, 254, 254, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2D  (32, 127, 127, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (32, 125, 125, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (32, 62, 62, 64)         0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (32, 60, 60, 64)          36928     
                                                                 
 max_pooling2d_2 (MaxPooling  (32, 30, 30, 64)         0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (32, 28, 28, 64)          36928     
                                                                 
 max_pooling2d_3 (MaxPooling  (32, 14, 14, 64)         0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (32, 12, 12, 64)          36928     
                                                                 
 max_pooling2d_4 (MaxPooling  (32, 6, 6, 64)           0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (32, 4, 4, 64)            36928     
                                                                 
 max_pooling2d_5 (MaxPooling  (32, 2, 2, 64)           0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (32, 256)                 0         
                                                                 
 dense (Dense)               (32, 64)                  16448     
                                                                 
 dense_1 (Dense)             (32, 3)                   195       
                                                                 
=================================================================
Total params: 183,747
Trainable params: 183,747
Non-trainable params: 0
_________________________________________________________________
[27]
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
[28]
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)
Epoch 1/100
29/29 [==============================] - 65s 1s/step - loss: 0.7691 - accuracy: 0.5017 - val_loss: 
[39]
history.history['accuracy']
[42]
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0) # Creating  a batche
    
    predictions = model.predict(img_array)
    
    predicted_class = class_names [np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence 
plt.figure(figsize=(25,25))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(model, images[i].numpy()) 
        actual_class = class_names[labels[i]]
        
        plt.title(f"Actual : {actual_class}, \n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        plt.axis("off")
[29]
model_version="v31"
model.save("D:/saved_M2_1/{model_version}")
WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 5 of 6). These functions will not be directly callable after loading.
INFO:tensorflow:Assets written to: D:/saved_M2_1/{model_version}\assets
INFO:tensorflow:Assets written to: D:/saved_M2_1/{model_version}\assets
[45]
model.summary()



[30]
import tensorflow as tf
#from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
[31]
model.save('D:/saved_M2_1/RC_2.h5')
print('Model Saved!')
Model Saved!
[32]
savedModel=load_model('D:/saved_M2_1/RC_2.h5')
savedModel.summary()

# Training the saved model again ...........
savedModel.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
[34]
history = savedModel.fit(
    train_ds,
    epochs=200,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)
Epoch 1/200
29/29 [==============================] - 21s 691ms/step - loss: 0.3930 - accuracy: 0.8306 - 
[35]
model_version="v32"
savedModel.save("D:/saved_M2_2/{model_version}")
WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 5 of 6). These functions will not be directly callable after loading.
INFO:tensorflow:Assets written to: D:/saved_M2_2/{model_version}\assets
INFO:tensorflow:Assets written to: D:/saved_M2_2/{model_version}\assets
[36]
savedModel.save('D:/saved_M2_2/RC_3.h5')
print('Model Saved!')
Model Saved!

