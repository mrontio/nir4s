import torch
from torch import nn
import norse
from norse.torch import SequentialState
import nir

# Norse
dt = 0.001

model = SequentialState(
    nn.Linear(100, 10, bias=False),
    norse.torch.LIFBoxCell(
        p=norse.torch.LIFBoxParameters(
            tau_mem_inv=torch.tensor([100.0]),
            v_th=torch.tensor([1.0]),
            v_reset=torch.tensor([0.0]),
            v_leak=torch.tensor([0.0]),
        )
    )
)
# Fill with increasing integers
model[0].weight.data[:] = torch.tensor(range(1, 100+1))

sample_data = torch.ones((100))
nir_graph = norse.torch.to_nir(torch.nn.Sequential(model), sample_data)
nir_graph.infer_types()
nir.write("network.nir", nir_graph)
