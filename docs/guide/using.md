# Using a NIRGraph

1. The NIRGraph imported in [Importing a NIRGraph](import.md) has the following structure
   ![LIF NIR Graph visualisation](img/lif.png "LIF NIR Graph visualisation")

2. In the ADT, this looks like
 ![LIF NIR Graph Scala ADT](img/lif-adt.png)
  **Notice**:
      1. They link 'backwards', top is Output node, bottom is Input node.
      2. The tensor types, documented in [Tensors](tensors.md), depend on whether we know the dimensions in advance or not.

3. Let's say we want to parse this into our own data structure. The general steps are
    1. Receive NIRNode `n`.
    2. Match against the final case class `NIRParam`s defined by api/NIRNode, which define the actual NIR node type.
    3. Match against the Static Tensor class defined in api/tensor/TensorStatic
    3. Transform this into your own data structure.

## Example procedure

1. Let's say we want to write a function `getVleak0` that returns the leaking voltage (`v_leak`) of all neuron modelin the graph, at *index 0*.
```scala
def getVleak0(n: NIRNode): List[Float] = {
    // ...
}
```
2. We match `NIRNode` against the ones we care about. In our example above, we use `Output`, `LIF` and `Input`.
```scala
n.params match {
    // case definitions for needed NIRNodes.x
}
```
3. `Input` is the Base case, as it has no previous nodes or weights, return an empty list.
```scala
case i: InputParams => List()
```
3. `LIFParam's` `v_leak` is defined as a `TensorStatic`, this means that we should match it against the different dimensions before we use it. Here, we just cover the 1D, 2D and 3D case, where we always access index 0.
```scala
case l: LIFParams => List(l.v_leak match {
  case t1d: Tensor1D[Float] => t1d(0)
  case t2d: Tensor2D[Float] => t2d(0, 0)
  case t3d: Tensor3D[Float] => t3d(0, 0, 0)
  case x            => throw new Exception()
})
```
4. Then, define the default case.
```scala
case _           => getVleak0(n.previous.head)
```
5. All together now, with calling code.
```scala
def getVleak0(n: NIRNode): List[Float] = {
  n.params match {
    case i: InputParams => List()
    case l: LIFParams => List(l.v_leak match {
      case t1d: Tensor1D[Float] => t1d(0)
      case t2d: Tensor2D[Float] => t2d(0, 0)
      case t3d: Tensor3D[Float] => t3d(0, 0, 0)
      case x            => throw new Exception()
    })
    case _           => getVleak0(n.previous.head)
  }
}

val network_path = "src/test/scala/nir/samples/lif/network.nir"
val g = NIRGraph(new File(network_path))
val result = getVleak0(g.output)
println(s"getVleak0 result = ${result}")
```
Sample output for this network would be `List(0.00023)`. This test can also be seen in [DocSpec](src/test/scala/nir/DocSpec.scala) test case.
