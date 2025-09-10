import torch
import norse
import nir

# Norse
dt = 0.001
li = norse.torch.LIBoxCell(p=norse.torch.LIBoxParameters(
            tau_mem_inv=torch.tensor([30.]),
            v_leak=torch.tensor([0.]),
        ), dt=dt)

sample_data = torch.ones((1))
nir_graph = norse.torch.to_nir(torch.nn.Sequential(li), sample_data)
nir_graph.infer_types()
nir.write("network.nir", nir_graph)
