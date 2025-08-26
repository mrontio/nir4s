import numpy as np
import tonic
import torch
import torch.nn as nn
from sinabs import from_model
from sinabs.nir import from_nir, to_nir

import nir

conv1d = nn.Sequential(
    nn.Conv1d(
        16, 16, kernel_size=3, stride=2, padding=1, bias=False
    ),
)

input_shape = (16, 20)
sample_data = torch.ones((1,) + input_shape)
snn = from_model(
    conv1d, input_shape=input_shape, batch_size=1, spike_threshold=1.0, min_v_mem=-1.0
).spiking_model

nir_graph = to_nir(snn, sample_data=sample_data)
nir_graph.infer_types()
nir.write("./network.nir", nir_graph)
