import numpy as np
import matplotlib.pyplot as plt

def plot_loss_accuracy_graphs(losses, accuracy):
    plt.figure(figsize=(12, 4))

    # Loss graph
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Loss Graph')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Accuracy graph
    plt.subplot(1, 2, 2)
    plt.plot(accuracy)
    plt.title('Accuracy Graph')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

def display_images_predictions(X, y_true, y_pred, num_images=5):
    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] #cifa10
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']  # mnist
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    axes = axes.ravel()
    indices = np.random.choice(len(X), num_images, replace=False)
    for i in range(num_images):
        index = indices[i]
        #axes[i].imshow(X[index].reshape(32, 32, 3), cmap=plt.cm.binary)# CIFAR10
        axes[i].imshow(X[index].reshape(28, 28), cmap=plt.cm.binary)# MNIST
        true_label = class_names[np.argmax(y_true[index])]
        pred_label = class_names[np.argmax(y_pred[index])]
        if true_label == pred_label:
            axes[i].set_title(f"True: {true_label} | Pred: {pred_label}", fontsize=12).set_color('green')
        else:
            axes[i].set_title(f"True: {true_label} | Pred: {pred_label}", fontsize=12).set_color('red')

        axes[i].axis('off')

    plt.subplots_adjust(wspace=0.5)
    plt.show()
