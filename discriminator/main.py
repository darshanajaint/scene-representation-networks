import torch
import argparse
from util import *
from ModelUtil import ModelUtil


def parse_args():
    parser = argparse.ArgumentParser("Configure the discriminator test.")
    parser.add_argument("--model_type", type=str,
                        help="Which model to use from 'mobilenet', 'resnet', "
                             "and 'googlenet'.")
    parser.add_argument("--validation_size", type=float,
                        help="Size of validation set split off from training "
                             "set.", default=0.1),
    parser.add_argument("--save_model_path", type=str,
                        help="Path to save the model after training.",
                        default="./model.pickle")
    parser.add_argument("--num_epochs", type=int,
                        help="The number of epochs to run the model for.",
                        default=10)
    return parser.parse_args()


def setup_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model_type)
    transform = get_transform()
    model = ModelUtil(model, transform, device, num_epochs=args.num_epochs)
    return model


if __name__ == "__main__":
    args = parse_args()
    model = setup_model(args)

    # Load data
    data, labels = load_data()
    train_data, val_data, train_labels, val_labels = train_val_split(
        data, labels, 1 - args.validation_size)

    # Train model
    train, val = model.train(train_data, train_labels, val_data, val_labels)

    # Evaluation metrics (loss, classification accuracy, balanced accuracy)
    save_model(args.save_model_path, train, val)
    plot_stats(train[0][0, :], val[0][0, :], 'Classification Accuracy')
    plot_stats(train[0][1, :], val[0][1, :], 'Balanced Accuracy')
    plot_stats(train[1], val[1], 'loss')

    # Other evaluation metrics
    val_pred, val_true = model.predict(val_data, val_labels)
    display_confusion_matrix(val_true, val_pred)
    calculate_precision(val_true, val_pred)
    calculate_recall(val_true, val_pred)
