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
val t1d: Tensor2D[Int] = ts match { t1d: Tensor2D => t1d; _ => println("Did not get 2D tensor!") }

td(0, 1)    // Compiles, runs.
td(1)       // Does not compile.
td(3, 1)    // Compiles, run-time error (TODO)!
```
