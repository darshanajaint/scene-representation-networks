# Todo:
#    - read all results files (for a single evaluation)
#    - skimage.metrics.peak_signal_to_noise_ratio

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from image_similarity_measures.quality_metrics import fsim

# psnr - true image, test image
# ssim - img1, img2
# fsim - orig image, pred image


def calculate_psnr(orig, test):
    return peak_signal_noise_ratio(orig, test)


def calculate_ssim(orig, test):
    return structural_similarity(orig, test)


def calculate_fsim(orig, test):
    return fsim(orig, test)

# box plots for each model and data set
# table of means for models and data sets, and pre gan training numbers
