import numpy as np
import tonic
import torch
import torch.nn as nn
from sinabs import from_model
from sinabs.nir import from_nir, to_nir

import nir

weight_shape = (16, 16, 3)
weight_size = weight_shape[0] * weight_shape[1] * weight_shape[2]
conv1d = nn.Conv1d(weight_shape[0], weight_shape[1], kernel_size=weight_shape[2], stride=2, padding=1, bias=False)

# Weight values = contiguous indices
weight = torch.linspace(0, weight_size, weight_size, dtype=torch.float).reshape(weight_shape)
conv1d.weight.data = weight.detach().clone().requires_grad_(True)

net = nn.Sequential(conv1d)

input_shape = (16, 20)
sample_data = torch.ones((1,) + input_shape)
snn = from_model(
    net, input_shape=input_shape, batch_size=1, spike_threshold=1.0, min_v_mem=-1.0
).spiking_model

nir_graph = to_nir(snn, sample_data=sample_data)
nir_graph.infer_types()
nir.write("./network.nir", nir_graph)
