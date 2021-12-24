from keras_preprocessing.image import ImageDataGenerator
from keras import layers, models
import tensorflow as tf

# 利用GPU参与训练
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)

# 读入图片集
train_dir = './Letter'
n_batch_size = 40

# 设置图片，将图片转换成机器可读形式
train_datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=n_batch_size, color_mode='grayscale',
    class_mode='categorical', subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=n_batch_size, color_mode='grayscale',
    class_mode='categorical', subset='validation')

# 生成器内容查看
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# 构建模型
model = models.Sequential()
model.add(layers.Conv2D(256, kernel_size=[3, 3], activation='relu', padding='same', strides=(1, 1),
                        input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2), strides=2, padding='valid'))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, kernel_size=[3, 3], activation='relu', padding='valid', strides=(1, 1),
                        input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2), strides=2, padding='valid'))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(26, activation='softmax'))

model.summary()

# # 编译模型
# model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
#
# # 训练模型
# history = model.fit_generator(train_generator, epochs=20, steps_per_epoch=len(train_generator),
#                               validation_data=validation_generator, validation_steps=len(validation_generator), verbose=1)
#
# model.save('./model1.h5')
