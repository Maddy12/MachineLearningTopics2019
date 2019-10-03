"""
This script is taken and modified from: https://github.com/eclique/RISE/blob/master/evaluation.py

@inproceedings{Petsiuk2018rise,
  title = {RISE: Randomized Input Sampling for Explanation of Black-box Models},
  author = {Vitali Petsiuk and Abir Das and Kate Saenko},
  booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
  year = {2018}
}

"""

from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import PIL.ImageOps as ops
import sys
import pdb
from myutils import datasets
import numpy as np
import sys
sys.path.append('/home/mschiappa/Desktop/MachineLearningTopics2019/pytorch-deeplab-xception')
from modeling.deeplab import *
from utils import *
from utils.metrics import Evaluator
# from explanations import RISE

# HW = 224 * 224 # image area
HW = 513*513
n_classes = 1000

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring.
    from: https://github.com/eclique/RISE/blob/master/evaluation.py

    """
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array.
    from: https://github.com/eclique/RISE/blob/master/evaluation.py

    """
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class PerturbeMetric():

    def __init__(self, model, step=1000):
        r"""Create deletion/insertion metric instance.
        from: https://github.com/eclique/RISE/blob/master/evaluation.py

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        self.model = model
        self.mode = 'del'
        self.step = step
        self.substrate_fn = torch.zeros_like
        # self.cat_list = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        # 1, 64, 20, 63, 7, 72]
        # self.cat_dict = {b:d for b, d in enumerate(self.cat_list)}

    def run(self, img_tensor, target, verbose=0, save_to=None):
        """Run metric on one image-saliency pair.
        Modified from: https://github.com/eclique/RISE/blob/master/evaluation.py

        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): mean output. 
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.
        Return:
            scores (nd.array): Array containing scores at every step.
        """
        # Get the prediction pixels by taking the max logit across the 21 classes
        n_samples = img_tensor.shape[0]
        pred, explanation = torch.max(self.model(img_tensor.cuda()), (1))  
        evaluator = Evaluator(21)
        # The number of steps is pixel by pixel, TODO is to change this to a set of pixels grouped around the chosen pixel
        n_steps = (HW + self.step - 1) // self.step  # HW is the area of the images, they made it a global variable....

        start = img_tensor.clone()  # original input 
        finish = self.substrate_fn(img_tensor)  # aim to end with all 0's 
        # miou = np.empty(n_steps + 1)
        miou = np.empty((n_samples, n_steps + 1))

        # Coordinates of pixels in order of decreasing saliency
        # orders the pixels from most important to least by providing the index 
        salient_order = np.flip(np.argsort(explanation.reshape(-1, HW).detach().cpu().numpy(), axis=1), axis=-1)  # the indexes of proper order
        r = np.arange(n_samples).reshape(n_samples, 1)
        for i in tqdm(range(n_steps+1), desc='Deleting Pixels'):
            pred, explanation = torch.max(self.model(start.cuda()), (1))
            for j in range(n_samples):
                evaluator.add_batch(target[j].detach().cpu().numpy(), explanation[j].detach().cpu().numpy())
                score = evaluator.Mean_Intersection_over_Union()
                evaluator.reset()    
                miou[j][i] = score
            if i < n_steps:
                # coords = salient_order[:, self.step * i:self.step * (i + 1)]
                # start.cpu().numpy().reshape(1, 3, HW)[0, :, coords] = finish.cpu().numpy().reshape(1, 3, HW)[0, :, coords]
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords] = finish.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords]
        robust_loss = [self.get_gradients(miou[x]) for x in range(n_samples)]   
        return robust_loss

    @staticmethod
    def get_gradients(miou):
        gradients = np.gradient(miou)
        idx = min(np.argsort(gradients)[:5]) # get top give drops
        # return abs(1.0/(gradients[idx] * float(idx+1)))
        return 1.0/float(idx)


if __name__ == "__main__":
    model_path = '/home/mschiappa/Desktop/MachineLearningTopics2019/pytorch-deeplab-xception/run/coco/deeplab-resnet/model_best.pth.tar'
    model = DeepLab(num_classes=21,
                    backbone="resnet",
                    output_stride=8,
                    sync_bn=None,
                    freeze_bn=False)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model = model.eval()
    model = model.cuda()
    train_loader, val_loader, test_loader, nclass = datasets.make_data_loader()
    for idx, sample in enumerate(val_loader):
        break
    # explanation = model(sample['image'][:1].cuda())
    # klen = 11
    # ksig = 5
    # kern = gkern(klen, ksig)

    # # Function that blurs input image
    # blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)
    deletion = PerturbeMetric(model, 513)
    print(deletion.run(sample['image'][:1], sample['label'][:1], verbose=1))