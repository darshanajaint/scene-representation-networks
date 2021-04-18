from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, images, labels, transform):
        """
            Args:
                images - iterable of all images
                labels - the true label for each image
                transform - the transform to apply to each image

            Modified from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        """
        self.images = images
        self.transform = transform
        self.img_list = [[self.transform(self.images[i]), labels[i]] for i in
                         self.images]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self.img_list[idx]
