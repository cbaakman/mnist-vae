import torch
import torchvision
from matplotlib import pyplot


def save_image(image: torch.Tensor, path: str):

    # convert 1-channel to 3-channel
    image = torch.concat((image, image, image), dim=0)

    pil = torchvision.transforms.ToPILImage()(image)

    pyplot.imshow(pil)

    pyplot.savefig(path)

def save_images_to_grid(images: torch.Tensor, path: str):

    grid_image = torchvision.utils.make_grid(images)

    pil = torchvision.transforms.ToPILImage()(grid_image)

    pyplot.imshow(pil)

    pyplot.savefig(path)

    pyplot.close()
