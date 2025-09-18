package nir.tensor

import io.jhdf.api.Dataset
import scala.reflect.ClassTag
import scala.language.implicitConversions


/* Dynamic Tensor
 * Dynamic construction of a tensor from runtime / unknown shape size.
 * Advantages: dynamic construction, reshaping, 1D backing array.
 * Disadvantages: Slower runtime due to checks, often requires asInstanceOf
 */


/** Utility to translate multidimensional tensor indices into a single
  * row-major offset.
  *
  * For a tensor of shape `(2, 3)` the index `(1, 2)` is computed as
  * \(1 \times 3 + 2 = 5\).
  *
  * @param shape dimensions of the tensor in each axis
  */
case class Indexer(shape: List[Int]) {

  def rank: Int = shape.length
  def size: Int = shape.reduce(_ * _)
  def rangeTree: RangeTree = RangeTree.buildTree(shape)

  private def idx(dims: List[Int]): Int = {
    // Row-major order (like numpy), (0, 0, 1) -> 1
    val shapeCumprod: List[Int] = shape.reverse.scanLeft(1)(_ * _).tail
    dims match {
      case Nil        => 0
      case dim :: Nil => dim
      case dim :: t   => idx(t) + dim * shapeCumprod(t.length - 1)
    }
  }

  def apply(vals: Seq[Int]): Int = {
    val vs = vals.toList
    // Rank check
    if (vs.length != rank) throw new IllegalArgumentException(s"Expected ${rank} indices but got ${vs.length}")

    // Bound check
    vs.zipWithIndex.foreach { case (v, i) =>
      val bound = shape(i)
      if (!(v >= 0 && v < bound)) throw new IndexOutOfBoundsException(s"Index $v out of bounds for axis $i (size $bound)")
    }

    // Run indexer
    idx(vs)
  }

  private def squeeze(shape: List[Int]): List[Int] =
    shape match {
      case Nil => Nil
      // case 1 :: Nil => Nil -- we assume we don't want to squeeze from the right
      case h :: Nil => h :: Nil
      case 1 :: t => squeeze(t)
      case h :: t => h :: squeeze(t)
    }

  def squeeze(): Indexer = {
    val ss = squeeze(shape)
    Indexer(ss)
  }

  def reshape(newshape: List[Int]): Indexer = {
    val newsize = newshape.reduce(_ * _)
    if (newsize != size) throw new IllegalArgumentException(s"New shape has size $newsize but expected $size.")

    Indexer(newshape)
  }
}

class TensorDynamic[D: ClassTag](data: Array[D], idx: Indexer) {
  val shape = idx.shape
  val rank = idx.rank
  val size = idx.size
  def apply(indices: Int*) = data(idx(indices))
  def reshape(newshape: Int*): TensorDynamic[D] = new TensorDynamic[D](data, idx.reshape(newshape.toList))
  def squeeze: TensorDynamic[D] = new TensorDynamic(data, idx.squeeze())
  def map[B: ClassTag](f: D => B): TensorDynamic[B] = new TensorDynamic[B](data.map(f), idx)

  private def toNestedList(tree: RangeTree): List[Any] = tree match {
    case Leaf(b, e) =>
      data.slice(b, e).toList

    case Branch(children) =>
      children.map(toNestedList)
  }

  def toList: List[_] = toNestedList(idx.rangeTree)

  // Map of Dynamic.Rank -> Static constructors
  private val staticConstructors: Map[Int, (Array[D], RangeTree) => TensorStatic[D]] = Map(
    1 -> Tensor1D.fromRangeTree[D],
    2 -> Tensor2D.fromRangeTree[D],
    3 -> Tensor3D.fromRangeTree[D],
    4 -> Tensor4D.fromRangeTree[D],
  )

  def toStatic: TensorStatic[D] = {
    staticConstructors.get(rank) match {
      case Some(constructor) => constructor(data, idx.rangeTree)
      case None => throw new Exception(s"Static tensors of dimension $rank are not yet supported.")
    }
  }
}

object TensorDynamic {
  // Allow to match based on rank
  object Rank {
    def unapply[D](tensor: TensorDynamic[D]): Option[Int] = Some(tensor.rank)
  }

  def flattenArray[T: ClassTag](a: Any): Any = {
    if (!a.isInstanceOf[Array[_]]) {
      throw new IllegalArgumentException("Input must be an array")
    }
    val flattened = flattenArrayRecursive(a).toArray
    flattened
  }

  private def flattenArrayRecursive(x: Any): Array[Any] = x match {
    case a: Array[_] => a.flatMap(flattenArrayRecursive)
    case v           => Array(v)
  }

  private def flattenDatasetArray[T: ClassTag](d: Dataset): Array[T] = {
    val rawData = d.getData // e.g. Array[Float], Array[Array[Float]], etc.
    val flat: Array[Any] = flattenArrayRecursive(rawData)
    flat.map(_.asInstanceOf[T]) // safe because caller controls T
  }

  // 1D case
  def apply[D: ClassTag](a: D*): TensorDynamic[D] = {
    val i = Indexer(List(a.length))
    new TensorDynamic[D](a.toArray, i)
  }

  // N-D case.
  def apply[D: ClassTag](a: Array[D], shape: List[Int]): TensorDynamic[D] = {
    val i = Indexer(shape)
    if (i.size != a.length) throw new IllegalArgumentException(s"Supplied array is size ${a.length} but shape is over size ${i.size}.")
    new TensorDynamic[D](a, i)
  }

  def apply[D: ClassTag](d: Dataset): TensorDynamic[D] = {
    val expectedType = implicitly[ClassTag[D]].runtimeClass
    val actualType = d.getDataType.getJavaType
    val idx = Indexer(d.getDimensions.toList)

    // Run-time checks
    require(expectedType != classOf[Nothing], s"TensorDynamic(Dataset) contructor requires type parameter [D].")
    require(actualType == expectedType, s"For dataset <${d.getName}>, expected type $expectedType but got $actualType.")

    new TensorDynamic(flattenDatasetArray[D](d), idx)
  }

  // Implicit conversions to Static
  implicit def toStaticConv[T](td: TensorDynamic[T]): TensorStatic[T] = td.toStatic





}
