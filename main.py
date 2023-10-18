import numpy as np
import tensorflow as tf
from MLP_leyer import MLP
from activation_loss_functions import CrossEntropy, one_hot_encoding, accuracy
from graphic_draw import display_images_predictions, plot_loss_accuracy_graphs


#(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data() #cifar10
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data() #mnist

y_train = one_hot_encoding(y_train)
X_train = X_train / 255.0

y_test = one_hot_encoding(y_test)
X_test = X_test / 255.0

#X_train = X_train.reshape(-1, 32 * 32 * 3) #cifar10
#X_test = X_test.reshape(-1, 32 * 32 * 3) #cifar10

X_train = X_train.reshape(-1, 28 * 28) #mnist
X_test = X_test.reshape(-1, 28 * 28) #mnist

X_train.shape, X_test.shape


#image_features = 32 * 32 * 3 #cifar10
#classes = 10 #cifar10
criterion = CrossEntropy()
classes = 10 #mnist
image_features = 28 * 28 #mnist
Net = MLP(image_features, classes)
learning_rate = 1e-4
epochs = 5
loss_grafic = []
accurency_grafic = []
batch_size = 128
n_samples = X_train.shape[0]

print("Start train...")
for epoch in range(epochs):
    print(f"\r[{'/' * (epoch+1)}{' ' * (epochs-epoch-1)}] Epoch {epoch+1}/{epochs}", end='')
    Loss = []
    Accuracy = []
    for i in range(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        x_batch, y_batch = X_train[begin:end], y_train[begin:end]

        Net_t = Net.forward(x_batch)
        Loss.append(np.mean(criterion.loss(y_batch, Net_t)))
        Accuracy.append(accuracy(np.argmax(y_batch, axis=1), np.argmax(Net_t, axis=1)))
        error = criterion.backward(y_batch, Net_t)
        Net.backward(error,learning_rate)

    print(f"  Loss: {np.mean(Loss)}, Acc: {np.mean(Accuracy)}")
    loss_grafic.append(np.mean(Loss))
    accurency_grafic.append(np.mean(Accuracy))

Net_t = Net.forward(X_test)
y_pred = np.argmax(Net_t, axis=1)  # predicted labels
y_true = np.argmax(y_test, axis=1)  # true labels
Acc_test  = accuracy(y_true, y_pred)
print("Test Accuracy:  ", Acc_test)

plot_loss_accuracy_graphs(loss_grafic, accurency_grafic)
display_images_predictions(X_test, y_test, Net_t)
