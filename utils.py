import matplotlib.pyplot as plt
from skimage.transform import pyramid_expand
from torch import cat, cuda, manual_seed, backends
from torch import device as torch_device
from torch.nn.utils.rnn import pad_sequence


# Constants
PATH = "flickr8k/"
NORMALISE = True
MODEL_PARAMS = {
    "vgg16": {
        "encoder_channels": 512,
        "features_dims": 7
    },
    "resnet50": {
        "encoder_channels": 2048,
        "features_dims": 8
    },
    "inception_v3": {
        "encoder_channels": 2048,
        "features_dims": 8
    }
}

MAGIC_MU = [0.485, 0.456, 0.406]
MAGIC_SIGMA = [0.229, 0.224, 0.225]

backends.cudnn.determinstic = True
if cuda.is_available():
    DEVICE = torch_device('cuda:0')
    cuda.manual_seed_all(42)
else:
    DEVICE = torch_device('cpu')
    manual_seed(42)


def collate(batch, pad_idx):
    images = [item[0].unsqueeze(0) for item in batch]
    images = cat(images, dim=0)

    captions = [item[1] for item in batch]
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)

    img_names = [item[2] for item in batch]
    return images, captions, img_names


def show_image(img, normalise, title=None):
    # Unnormalise
    if normalise:
        for i in range(3):
            img[i] *= MAGIC_SIGMA[i]
            img[i] += MAGIC_MU[i]

    img = img.numpy().transpose((1, 2, 0))

    if title is not None:
        plt.title(title)
    plt.imshow(img)
    #plt.pause(1e-3)
    #plt.show()

def plot_attention(img, caption, alphas, normalise=False, features_dims=7):
    # Unnormalise
    if normalise:
        for i in range(3):
            img[i] *= MAGIC_SIGMA[i]
            img[i] += MAGIC_MU[i]

    img = img.numpy().transpose((1, 2, 0))
    img_cpy = img

    fig = plt.figure(figsize=(15, 15))

    len_caption = len(caption)
    for i in range(len_caption):
        att = alphas[i].reshape(features_dims, features_dims)
        att = pyramid_expand(att, upscale=24, sigma=8)

        ax = fig.add_subplot(len_caption // 2, len_caption // 2, i + 1)
        ax.set_title(caption[i])
        img = ax.imshow(img_cpy)
        ax.imshow(att, cmap='gray', alpha=0.7, extent=img.get_extent())

    plt.tight_layout()
    plt.show()
