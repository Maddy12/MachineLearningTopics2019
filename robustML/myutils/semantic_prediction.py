import torch


def pred(output_sample, visualize=False):
    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]
    values, indice = torch.max(output)
    if visualize: 
        plt.imshow(indice.detach().cpu().numpy(), cmap='jet')
        plt.show()
    return values, indice



def tensor_imshow(inp, title=None, **kwargs):
    """
    Imshow for Tensor.
    From https://github.com/eclique/RISE/blob/master/utils.py
    """
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


def view_output_heatmap(output_class, avg=False):
    if avg:
        plt.imshow(torch.mean(output_class, (0)).detach().cpu().numpy(), cmap='jet'); plt.show()
    else:
        output_class = output_class.detach().cpu().numpy()
        inverted_image = PIL.ImageOps.invert(Image.fromarray(np.uint8(output_class)))
        plt.axis('off')
        plt.imshow(inverted_image, cmap='jet')
        plt.show()


def vec_translate(a, d):
    """
    Args:
        a (array)
        d (dict)
    returns the actual class, justy kidding don't need this but I spent an hour so keeping it :'(
    """
    return np.vectorize(d.__getitem__)(a)
