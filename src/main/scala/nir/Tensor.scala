package nir

import io.jhdf.api.Dataset
import scala.reflect.ClassTag

sealed trait RangeTree
case class Leaf(begin: Int, end: Int) extends RangeTree
case class Branch(children: List[RangeTree]) extends RangeTree

class Counter(var value: Int) {
  def next(): Int = { val cur = value; value += 1; cur }
  def plus(x: Int): (Int, Int) = { val cur = value; value += x; (cur, value) }
}

case class Indexer(shape: List[Int]) {
  def rank: Int = shape.length
  def size: Int = shape.reduce(_ * _)
  def rangeTree: RangeTree = buildTree(shape)

  def buildTree(shape: List[Int]): RangeTree = {
    val counter = new Counter(0)

    def build(dims: List[Int]): RangeTree = dims match {
      case Nil =>
        val idx = counter.next()
        Leaf(idx, idx + 1)

      case dim :: Nil =>
        val (b, e) = counter.plus(dim)
        Leaf(b, e)

      case h :: t =>
        val children = (0 until h).map(_ => build(t)).toList
        Branch(children)
    }

    build(shape)
  }

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
    if (vs.length != rank) throw new IllegalArgumentException(s"Expected ${rank} indexes but got ${vs.length}")

    vs.zipWithIndex.foreach { case (v, i) =>
      val bound = shape(i)
      if (!(v >= 0 && v < bound)) throw new IndexOutOfBoundsException(s"Index $v out of bounds for axis $i (size $bound)")
     }

    idx(vs)
  }

  private def squeeze(shape: List[Int]): List[Int] =
    shape match {
      case Nil => Nil
      case 1 :: Nil => Nil
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

/* =======================
 *   Sealed Tensor trait
 * ======================= */
sealed trait Tensor[D] {
  def data: Array[D]
  def idx: Indexer

  final def shape: List[Int] = idx.shape
  final def rank: Int        = idx.rank
  final def size: Int        = idx.size

  def apply(indices: Int*): D = data(idx(indices))

  /** Map over elements; preserves shape and returns best-fit subtype by rank */
  def map[B: ClassTag](f: D => B): Tensor[B] =
    Tensor.from(data.map(f), idx)

  /** Reshape; returns best-fit subtype by rank */
  def reshape(newshape: List[Int]): Tensor[D] =
    Tensor.from(data, idx.reshape(newshape))

  protected def toNestedList(tree: RangeTree): List[_] = tree match {
    case Leaf(b, e)      => data.slice(b, e).toList
    case Branch(children) => children.map(toNestedList)
  }

  /** Generic (weakly typed) nested list */
  def toList: List[_] = toNestedList(idx.rangeTree)
}

/* -------------------------------------------------
 * Base implementation to share logic across subtypes
 * ------------------------------------------------- */
abstract class BaseTensor[D](val data: Array[D], val idx: Indexer) extends Tensor[D]

object Tensor {
  /** Factory that picks the most precise subtype by rank (1..5), else generic ND */
  def from[D](a: Array[D], i: Indexer): Tensor[D] = i.rank match {
    case 1 => new Tensor1D(a, i)
    case 2 => new Tensor2D(a, i)
    case 3 => new Tensor3D(a, i)
    case 4 => new Tensor4D(a, i)
    case 5 => new Tensor5D(a, i)
    case x => throw new Exception(s"Tensor of rank $x not yet supported")
  }

  /** Public constructor from (data, shape) */
  def apply[D](a: Array[D], shape: List[Int]): Tensor[D] = {
    val i = Indexer(shape)
    if (i.size != a.length)
      throw new IllegalArgumentException(
        s"Supplied array is size ${a.length} but shape is over size ${i.size}."
      )
    from(a, i)
  }

  // Extractor to match by rank
  object Rank {
    def unapply[D](tensor: Tensor[D]): Option[Int] = Some(tensor.rank)
  }

  // Dataset loading helpers
  private def flattenArrayRecursive(x: Any): Array[Any] = x match {
    case a: Array[_] => a.flatMap(flattenArrayRecursive)
    case v           => Array(v)
  }

  private def flattenDatasetArray[T: ClassTag](d: Dataset): Array[T] = {
    val rawData: Any       = d.getData
    val flat: Array[Any]   = flattenArrayRecursive(rawData)
    flat.map(_.asInstanceOf[T]) // caller controls T via d.getDataType
  }

  def apply(d: Dataset): Tensor[_] = {
    val idx = Indexer(d.getDimensions.toList)
    d.getDataType.getJavaType.toString match {
      case "float" => from(flattenDatasetArray[Float](d), idx)
      case "int"   => from(flattenDatasetArray[Int](d),   idx)
      case "long"  => from(flattenDatasetArray[Long](d),  idx)
      case o       => throw new java.text.ParseException(
        s"Java type \"$o\" not yet supported for Tensor conversion", 0
      )
    }
  }
}

/*
 PATTERN MATCHING
 ----------------

 All the above logic serves to create these final classes.
 You can pattern match against them, importantly they don't require
 the use of asInstanceOf for functions (e.g. toList).
 This is a work in progress.

 */
final class Tensor1D[D](override val data: Array[D], override val idx: Indexer)
  extends BaseTensor[D](data, idx) {
  require(idx.rank == 1, s"Expected rank 1 tensor, got ${idx.rank}")
  override def toList: List[D] =
    toNestedList(idx.rangeTree).asInstanceOf[List[D]]
}

object Tensor1D {
  def apply[D](data: Array[D], idx: Indexer): Tensor1D[D] =
    new Tensor1D(data, idx)

  def unapply[D](t: Tensor[D]): Option[Tensor1D[D]] =
    if (t.idx.rank == 1) Some(new Tensor1D(t.data, t.idx)) else None
}

final class Tensor2D[D](override val data: Array[D], override val idx: Indexer)
  extends BaseTensor[D](data, idx) {
  require(idx.rank == 2, s"Expected rank 2 tensor, got ${idx.rank}")
  override def toList: List[List[D]] =
    toNestedList(idx.rangeTree).asInstanceOf[List[List[D]]]
}

object Tensor2D {
  def apply[D](data: Array[D], idx: Indexer): Tensor2D[D] =
    new Tensor2D(data, idx)

  def unapply[D](t: Tensor[D]): Option[Tensor2D[D]] =
    if (t.idx.rank == 2) Some(new Tensor2D(t.data, t.idx)) else None
}

final class Tensor3D[D](override val data: Array[D], override val idx: Indexer)
  extends BaseTensor[D](data, idx) {
  require(idx.rank == 3, s"Expected rank 3 tensor, got ${idx.rank}")
  override def toList: List[List[List[D]]] =
    toNestedList(idx.rangeTree).asInstanceOf[List[List[List[D]]]]
}

object Tensor3D {
  def apply[D](data: Array[D], idx: Indexer): Tensor3D[D] =
    new Tensor3D(data, idx)

  def unapply[D](t: Tensor[D]): Option[Tensor3D[D]] =
    if (t.idx.rank == 3) Some(new Tensor3D(t.data, t.idx)) else None
}

final class Tensor4D[D](override val data: Array[D], override val idx: Indexer)
  extends BaseTensor[D](data, idx) {
  require(idx.rank == 4, s"Expected rank 4 tensor, got ${idx.rank}")
  override def toList: List[List[List[List[D]]]] =
    toNestedList(idx.rangeTree).asInstanceOf[List[List[List[List[D]]]]]
}

object Tensor4D {
  def apply[D](data: Array[D], idx: Indexer): Tensor4D[D] =
    new Tensor4D(data, idx)

  def unapply[D](t: Tensor[D]): Option[Tensor4D[D]] =
    if (t.idx.rank == 4) Some(new Tensor4D(t.data, t.idx)) else None
}

final class Tensor5D[D](override val data: Array[D], override val idx: Indexer)
  extends BaseTensor[D](data, idx) {
  require(idx.rank == 5, s"Expected rank 5 tensor, got ${idx.rank}")
  override def toList: List[List[List[List[List[D]]]]] =
    toNestedList(idx.rangeTree).asInstanceOf[List[List[List[List[List[D]]]]]]
}

object Tensor5D {
  def apply[D](data: Array[D], idx: Indexer): Tensor5D[D] =
    new Tensor5D(data, idx)

  def unapply[D](t: Tensor[D]): Option[Tensor5D[D]] =
    if (t.idx.rank == 5) Some(new Tensor5D(t.data, t.idx)) else None
}
