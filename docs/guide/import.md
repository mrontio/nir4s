# Importing a NIR Graph

1. You have a NIR Graph exported from Python, such as
```python
import torch
import norse
import nir

# Norse
dt = 0.001
lif = norse.torch.LIFBoxCell(p=norse.torch.LIFBoxParameters(
    tau_mem_inv=torch.tensor([50.]),
    v_th=torch.tensor([1.]),
    v_reset=torch.tensor([0.]),
    v_leak=torch.tensor([0.]),
))

sample_data = torch.ones((100, 1))
nir_graph = norse.torch.to_nir(torch.nn.Sequential(lif), sample_data)
nir.write("network.nir", nir_graph)
```

2. In your Scala project, import the network with
```scala
import nir.{NIRGraph}

val g = NIRGraph(new File("./network.nir"))
```
3. That's it! See examples of usage in [Using a NIRGraph](using.md)
