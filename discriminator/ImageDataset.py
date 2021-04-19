from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, reals, fakes, transform):
        """
            Args:
                images - iterable of all images
                labels - the true label for each image
                transform - the transform to apply to each image

            Modified from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        """
        self.reals = reals
        self.fakes = fakes
        self.transform = transform

        self.img_list = []
        for i in range(len(self.reals)):
            print("Before, ", self.reals[i].shape)
            real = self.transform(self.reals[i])
            fake = self.transform(self.fakes[i])
            print("After, ", real.shape)
            self.img_list.append([real, fake])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self.img_list[idx]
