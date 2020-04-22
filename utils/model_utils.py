import numpy as np
import torch.nn as nn


# Layer Initialization
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def weights_init(model):
    if isinstance(model, nn.Linear):
        print(model)
        model.weight.data.uniform_(*hidden_init(model))