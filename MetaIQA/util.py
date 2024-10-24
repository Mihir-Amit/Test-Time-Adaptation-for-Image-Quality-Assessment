import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import  Line2D
from PIL import Image
from torchvision import transforms
from PIL import ImageCms
from skvideo.utils.mscn import gen_gauss_window
import scipy.ndimage

import lpips

class GroupContrastiveLoss(nn.Module):
	def __init__(self, batch_size, temperature=0.5):
		super().__init__()
		self.batch_size = batch_size
		self.register_buffer("temperature", torch.tensor(temperature))
		self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
		self.register_buffer("positives_mask", (~torch.eye(batch_size * 1, batch_size * 1, dtype=bool)).float())

	def forward(self, emb_i, emb_j):
		"""
		emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
		z_i, z_j as per SimCLR paper
		"""

		self.negatives_mask[:len(emb_i), :len(emb_j)]=False
		self.negatives_mask[len(emb_i):, len(emb_j):] = False

		z_i = F.normalize(emb_i, dim=1)
		z_j = F.normalize(emb_j, dim=1)

		representations = torch.cat([z_i, z_j], dim=0)
		similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

		pos_similarity_matrix = similarity_matrix[:len(emb_i), :len(emb_j)]
		neg_similarity_matrix = similarity_matrix[len(emb_i):, len(emb_j):]

		pos_similarity_matrix = pos_similarity_matrix * self.positives_mask
		sim_ij=torch.sum(pos_similarity_matrix,dim=1)/(len(neg_similarity_matrix)-1)

		neg_similarity_matrix = neg_similarity_matrix * self.positives_mask
		sim_ji = torch.sum(neg_similarity_matrix, dim=1)/(len(neg_similarity_matrix)-1)

		positives = torch.cat([sim_ij, sim_ji], dim=0)

		nominator = torch.exp(positives / self.temperature)
		denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

		loss_partial = torch.sum(nominator / (nominator + torch.sum(denominator, dim=1)))/ (2 * self.batch_size)
		loss = -torch.log(loss_partial)

		return loss

class ContrastiveLoss(nn.Module):
	def __init__(self, batch_size, temperature=0.5):
		super().__init__()
		self.batch_size = batch_size
		self.register_buffer("temperature", torch.tensor(temperature))
		self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

	def forward(self, emb_i, emb_j):
		"""
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
		z_i = F.normalize(emb_i, dim=1)
		z_j = F.normalize(emb_j, dim=1)

		representations = torch.cat([z_i, z_j], dim=0)
		similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

		sim_ij = torch.diag(similarity_matrix, self.batch_size)
		sim_ji = torch.diag(similarity_matrix, -self.batch_size)
		positives = torch.cat([sim_ij, sim_ji], dim=0)

		nominator = torch.exp(positives / self.temperature)
		denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

		loss_partial = -torch.log(nominator / (nominator+torch.sum(denominator, dim=1)))
		loss = torch.sum(loss_partial) / (2 * self.batch_size)

		return loss

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

def ResizeCrop(image, sz, div_factor):
    image_size = image.size
    image = transforms.Resize([image_size[1] // div_factor, \
                               image_size[0] // div_factor])(image)

    if image.size[1] < sz[0] or image.size[0] < sz[1]:
        # image size smaller than crop size, zero pad to have same size
        image = transforms.CenterCrop(sz)(image)
    else:
        image = transforms.RandomCrop(sz)(image)

    return image


def compute_MS_transform(image, window, extend_mode='reflect'):
    h, w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    scipy.ndimage.correlate1d(image, window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, window, 1, mu_image, mode=extend_mode)
    return image - mu_image


def MS_transform(image):
    #   MS Transform
    image = np.array(image).astype(np.float32)
    window = gen_gauss_window(3, 7 / 6)
    image[:, :, 0] = compute_MS_transform(image[:, :, 0], window)
    image[:, :, 0] = (image[:, :, 0] - np.min(image[:, :, 0])) / (np.ptp(image[:, :, 0]) + 1e-3)
    image[:, :, 1] = compute_MS_transform(image[:, :, 1], window)
    image[:, :, 1] = (image[:, :, 1] - np.min(image[:, :, 1])) / (np.ptp(image[:, :, 1]) + 1e-3)
    image[:, :, 2] = compute_MS_transform(image[:, :, 2], window)
    image[:, :, 2] = (image[:, :, 2] - np.min(image[:, :, 2])) / (np.ptp(image[:, :, 2]) + 1e-3)

    image = Image.fromarray((image * 255).astype(np.uint8))
    return

def colorspaces(im, val):
    if val == 0:
        im = transforms.RandomGrayscale(p=1.0)(im)
    elif val == 1:
        srgb_p = ImageCms.createProfile("sRGB")
        lab_p  = ImageCms.createProfile("LAB")

        rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
        im = ImageCms.applyTransform(im, rgb2lab)
    elif val == 2:
         im = im.convert('HSV')
    elif val == 3:
         im = MS_transform(im)
    return im


def Sort_Tuple(tup):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    tup.sort(key=lambda x: x[1],reverse=True)
    return tup

df = pd.DataFrame()

def append_to_dataframe(layer_grad, layer_avg_grad, layer_name, similarity_exceeds_threshold, cosine_similarity_value):
    global df  # Use the global DataFrame to accumulate data

    # Convert the gradients to lists (assuming layer_grad and layer_avg_grad are tensors or arrays)
    layer_grad_np = layer_grad.detach().cpu().numpy() if hasattr(layer_grad, 'detach') else np.array(layer_grad)
    layer_avg_grad_np = layer_avg_grad.detach().cpu().numpy() if hasattr(layer_avg_grad, 'detach') else np.array(layer_avg_grad)
    
    # Create a DataFrame for the current layer
    df_iteration = pd.DataFrame({
        'layer_name': [layer_name],  # Add layer name
        'layer_grad': [layer_grad_np.tolist()],  # Store gradients as lists
        'layer_avg_grad': [layer_avg_grad_np.tolist()],  # Store average gradients as lists
        'cosine_similarity': [cosine_similarity_value],  # Store cosine similarity
        'similarity_exceeds_threshold': [similarity_exceeds_threshold]  # True/False value
    })

    # Append a row of NaN values for spacing between iterations
    df_iteration = df_iteration.append(pd.Series([None] * len(df_iteration.columns), index=df_iteration.columns), ignore_index=True)

    # Append the current iteration's DataFrame to the global DataFrame
    global df
    df = pd.concat([df, df_iteration], ignore_index=True)
    # print(df.iloc[-5])

def add_empty_line():
    global df  # Use the global DataFrame to append the empty line
    # Create an empty row with None values
    empty_row = pd.Series([None] * len(df.columns), index=df.columns)
    # Append the empty row to the global DataFrame
    df = df.append(empty_row, ignore_index=True)

def save_to_parquet(filename='gradients_with_similarity.parquet'):
    global df
    # Save the DataFrame to a Parquet file
    df.to_parquet(filename, engine='pyarrow')

def lpips_losses(image):
    ref_image = Image.open('..Datasets/References/refimg')

    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    # Define a transform to resize the image to (264, 264) and convert to a tensor
    transform = transforms.Compose([
        transforms.Resize((264, 264)),     # Resize to 264x264
        transforms.ToTensor()              # Convert image to Tensor (automatically scales pixels to [0,1])
    ])
    image_tensor = transform(image)
    ref_image_tensor = transform(ref_image)

    image_tensor = image_tensor*2-1
    ref_image_tensor = ref_image_tensor*2-1

    if lpips(image_tensor, ref_image_tensor) < 0.5:
        return True
    else:
        return False
         
