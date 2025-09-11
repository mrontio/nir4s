# Tensor Implementation

We need two distinct Tensor implementations to have both flexibility during interpretation and guarantees during use. This is encoded with our two Tensor abstractions:


| TensorDynamic                              | TensorStatic                     |
|--------------------------------------------|----------------------------------|
| Aims for run-time safety.                  | Aims for compile-time safety.    |
| Mutable data.                              | Immutable data.                  |
| Mutable shape (squeeze, reshape, flatten). | Immutable shape.                 |
| Pattern matched based on run-time Rank.    | Pattern matched on compile type  |
| 1D storage array + indexer.                | Inductively defined.             |
| Conclusion: use for reading in.            | Conclusion: export from dynamic. |

**Why?** We can't predict anything the NIR graph, which has to be runtime-checked anyway, whilst you get a compile-time safe interface to reading NIR files.

## Comparison by example
### Creating
- TensorDynamic
```scala
// Specify storage array and shape of it.
val td: TensorDynamic[Int] = new TensorDynamic(List(1, 2, 3, 4, 5, 6), List(2, 3))
println(td)
```
- TensorStatic
```scala
// TensorStatic can be created from TensorDynamic
val ts: TensorStatic[Int] = new TensorDynamic(List(1, 2, 3, 4, 5, 6), List(2, 3))
println(ts)
```
### Accessing
- TensorDynamic
```scala
val td: TensorDynamic[Int] = new TensorDynamic(List(1, 2, 3, 4, 5, 6), List(2, 3))
td(0, 1)    // Compiles, runs.
td(1)       // Compiles, run-time error!
td(3, 1)    // Compiles, run-time error!
```
- TensorStatic
```scala
val ts: TensorStatic[Int] = new TensorDynamic(List(1, 2, 3, 4, 5, 6), List(2, 3))
// Must be matched!
val t2d: Tensor2D[Int] =
    ts match {
        case t2d: Tensor2D => t2d
        case_              => throw new Exception("Rank not supported.")
    }

td(0, 1)    // Compiles, runs.
td(1)       // Does not compile.
td(3, 1)    // Compiles, run-time error (TODO)!
```

## Use within NIRNodes
Sometimes, the shape for a specific NIR parameter is well-defined.

For example, we know that the `Input` node's `shape` praameter will always be 1-dimensional. Hence, we define it as
```scala
final case class InputParams(
  shape: Tensor1D[Long],
) extends NIRParams
```
In nir4s, this serves as a template for acceptable types.

Other times, you don't know the shape of your parameters.

For example, if the preceding node is `Affine` (dense connection), we know our next `LIF` layer node will have a 1D `tau`, `r`, `v_leak` and `v_threshold`.

However, what if the preceding layer is a `Conv2d` node? Now all the parameters are of dimension 3. As we cannot predict this, we specify the looser (but still statically type-checked) `TensorStatic`.
```scala
final case class LIFParams(
  tau: TensorStatic[Float],
  r: TensorStatic[Float],
  v_leak: TensorStatic[Float],
  v_threshold: TensorStatic[Float],
) extends NIRParams
```
This can then be matched against, as in the "Accessing" section above.
