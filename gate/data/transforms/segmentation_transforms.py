import random


class SameRandomCrop:
    def __init__(self, output_size):
        """
        :param output_size: Desired output size of the crop. Either int or Tuple.
        """
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, img1, img2):
        """
        :param img1: First image to be cropped.
        :param img2: Second image to be cropped.
        :return: Tuple of cropped images.
        """
        if img1.shape[-2:] != img2.shape[-2:]:
            raise ValueError("Input images must have the same dimensions!")

        h, w = img1.shape[-2:]
        new_h, new_w = self.output_size

        # Generate random coordinates for the top left corner of the crop
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        # Crop the images using the generated coordinates
        img1 = img1[:, top : top + new_h, left : left + new_w]
        img2 = img2[:, top : top + new_h, left : left + new_w]

        return img1, img2
