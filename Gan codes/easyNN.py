import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.random.normal([1, 100]))


# Descarreguem les imatges de la base de dades
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0


# plot de imatges a la base de dades
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)   # el segon argument fa que surtin en blanc i negre
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


# La xarxa
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=25)
model.save('./juguete.h5')


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
print('\nTest accuracy:', test_acc, '\n')

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)


def plotimage(im):
    print('\n Image number', im)
    arg = np.argmax(predictions [im])
    #print(class_names[arg],predictions[im][arg])
    for i in range(10):
        print(class_names[i], " :", predictions[im][i])
    plt.figure()
    plt.imshow(test_images[im],cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()


plotimage(13)
plotimage(14)
plotimage(15)