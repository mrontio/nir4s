import torch
import torch.nn as nn
import numpy as np
from sinabs import from_model, layers
import sinabs.activation.spike_generation as spikegen
from sinabs.nir import to_nir
import nir


# Create the network in torch
ann = nn.Sequential(
    nn.Conv2d(2, 16, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
    nn.ReLU(),
    layers.SumPool2d(kernel_size=(2,2)),
    nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
    nn.ReLU(),
    layers.SumPool2d(kernel_size=(2,2)),
    nn.Flatten(),
    nn.Linear(4 * 4 * 8, 256, bias=False),
    nn.ReLU(),
    nn.Linear(256,10, bias=False),
    nn.ReLU(),
)

# Load weights and data sample
sample = np.load('./sample.npy')
ann.load_state_dict(torch.load('./weights.pth'))

# Convert to sinabs
device = 'cpu'
sample = torch.Tensor(sample).to(device)
sample = sample.reshape(10, 2, 34, 34)
snn = from_model(
    ann,
    input_shape=sample.shape[1:],
    num_timesteps=10,
    batch_size=1,
    spike_threshold=1.0,
    spike_fn = spikegen.SingleSpike,
    min_v_mem=-1.0
).spiking_model

# Save as nir file
snn_nir = to_nir(snn, sample_data=sample)
snn_nir.infer_types()
nir.write("./network.nir", snn_nir)
