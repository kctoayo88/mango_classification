  
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.applications.xception import Xception
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from efficientnet.keras import EfficientNetB3
import numpy as np

mcp_save = ModelCheckpoint('./models/model.h5', save_best_only=True, monitor='val_loss', mode='min')

img_width, img_height = 224, 224
n_batch_size = 16
n_epochs = 500
n_training_steps_per_epoch = 3000
n_validation_steps_per_epoch = 300

train_datagen = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=True,
                                   validation_split=0.2,
                                   rescale=1./255.)

train_generator = train_datagen.flow_from_directory('./train/data', 
                                                    target_size=(img_height, img_width), 
                                                    batch_size=n_batch_size,
                                                    shuffle=True,
                                                    subset='training')

vali_generator = train_datagen.flow_from_directory('./train/data',
                                                   target_size=(img_height, img_width),
                                                   batch_size=n_batch_size,
                                                   shuffle=True,
                                                   subset='validation')

# 以訓練好的 EfficientNetB3 為基礎來建立模型
net = EfficientNetB3(input_shape=(img_height, img_width, 3),
                     include_top=False, 
                     weights='imagenet', 
                     pooling='max')

# 增加 Dense layer 與 softmax 產生個類別的機率值
x = net.output
# x = Dense(1024,  activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(128,  activation='relu')(x)
output_layer = Dense(3, activation='softmax', name='softmax')(x)

# 設定要進行訓練的網路層
model = Model(inputs=net.input, outputs=output_layer)

print('\n')
print('Trainable layers:')
for x in model.trainable_weights:
    print(x.name)

model.compile(optimizers.Adam(lr=1e-3), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
print('\n')

# 輸出整個網路結構
model.summary()

model.fit_generator(train_generator,
                    steps_per_epoch = n_training_steps_per_epoch,
                    validation_data = vali_generator,
                    epochs = n_epochs,
                    validation_steps = n_validation_steps_per_epoch,
                    class_weight='balanced',
                    callbacks = [mcp_save])

model.save('./models/final_model.h5')