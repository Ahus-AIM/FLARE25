import random

import torch
import torch.nn.functional as F


class CropOrPad(torch.nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor, seed):
        # ensure the input has shape target_shape in the last 3 dimensions
        input_shape = tensor.shape[-3:]
        if input_shape == self.target_shape:
            return tensor
        if input_shape[0] < self.target_shape[0]:
            tensor = F.pad(tensor, (0, 0, 0, 0, 0, self.target_shape[0] - input_shape[0]))
        if input_shape[1] < self.target_shape[1]:
            tensor = F.pad(tensor, (0, 0, 0, self.target_shape[1] - input_shape[1], 0, 0))
        if input_shape[2] < self.target_shape[2]:
            tensor = F.pad(tensor, (0, self.target_shape[2] - input_shape[2], 0, 0, 0, 0))

        return tensor


class Flip(torch.nn.Module):
    def __init__(self):
        """
        Randomly transpose the input tensor along one of the three possible axes.

        Additionally, randomly flip a dimension of the tensor.
        """
        super().__init__()
        self.transpose_dims = [[-3, -1, -2], [-1, -2, -3], [-2, -3, -1]]
        self.flip_dims = [-1, -2, -3]

    def forward(self, tensor, seed):
        random.seed(seed)
        if random.random() < 0.5:
            flip = random.choice(self.flip_dims)
            tensor = torch.flip(tensor, [flip])
        if random.random() < 0.5:
            tensor = tensor.permute(*random.choice(self.transpose_dims))
        return tensor


class Compose(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, tensor, seed):
        for transform in self.transforms:
            tensor = transform(tensor, seed)
        return tensor
