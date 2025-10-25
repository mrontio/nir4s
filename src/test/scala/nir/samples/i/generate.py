import numpy as np
import nir

r_values = np.array([1.0])  # Example resistance values
integrator = nir.ir.I(r=r_values)

nir_model = nir.NIRGraph(
    nodes={
        "input": nir.Input(input_type=np.array([1])),
        "i": integrator,
        "output": nir.Output(output_type=np.array([1])),
    },
    edges=[("input", "i"), ("i", "output")],
)

nir_model.infer_types()
nir.write("network.nir", nir_model)
