import matplotlib.pyplot as plt

from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from discriminator.models import MobileNetModel, ResNetModel, GoogLeNetModel
from pickle import dump


def get_transform():
    return transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        # transforms.ToTensor()
    ])


def train_val_split(data, labels, train_size=0.9):
    return train_test_split(data, labels, train_size=train_size)


def get_model(model_type):
    if model_type == "mobilenet":
        return MobileNetModel()
    elif model_type == "resnet":
        return ResNetModel()
    elif model_type == "googlenet":
        return GoogLeNetModel()
    else:
        raise ValueError("model_type must be one of 'mobilenet', 'resnet', "
                         "and 'googlenet'.")


def save_model(path, train, val):
    state = {
        'training_accuracy': train[0],
        'training_loss': train[1],
        'validation_accuracy': val[0],
        'validation_loss': val[1]
    }
    with open(path, 'wb') as file:
        dump(state, file)


def display_confusion_matrix(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))


def calculate_precision(y_true, y_pred):
    print("Precision:", precision_score(y_true, y_pred))


def calculate_recall(y_true, y_pred):
    print("Recall:", recall_score(y_true, y_pred))


def plot_stats(train, val, stat_type):
    plt.figure(figsize=(10, 8))

    epochs = list(range(len(train)))
    stat_type = stat_type.capitalize()
    train_label = 'Training ' + stat_type
    val_label = 'Validation ' + stat_type

    plt.plot(epochs, train, c='b', label=train_label)
    plt.plot(epochs, val, c='g', label=val_label)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel(stat_type)
    plt.title("Plot of Training and Validation " + stat_type)

    # TODO: add a path arg to this function and save the figure.
    #   plt.savefig(path)


def load_data():
    # TODO: write image loading function.
    return None
