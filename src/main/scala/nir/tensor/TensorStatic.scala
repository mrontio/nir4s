package nir.tensor

import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag
import shapeless.HList

/* Static Tensor
 * Immutable Tensor class with safety guarantees.
 * Isomorphic with TensorDynamic (typically constructed from it)
 * Advantages: Known compile-time types for all functions, faster runtime.
 * Disadvantages: No reshaping, non-arbitrary dimensions.
 */

sealed trait TensorStatic[T] {
  val shape: List[Int]
  def rank: Int
  def length: Int
  def size: Int

  def toString: String

  /* TODO:
   * apply: User-facing, needs type-checking.
   * at: internal, just access.
   */
  protected def at(idx: Seq[Int]): T
  def apply(idx: Int*): T

  def map[B: ClassTag](f: T => B): TensorStatic[B]

  // Ambiguous type will be over-written by children
  def toList: List[_]

 class ShapeException(message: String) extends IllegalArgumentException(message)
}

case class Tensor1D[T](data: Array[T], shape: List[Int]) extends TensorStatic[T] {
  // TODO: Somehow make this compile-safe, _without_ writing 500 lines of code.
  require(shape.length == 1)

  override def rank: Int = 1
  override def length: Int = data.length
  override def size: Int = data.length

  override def map[B: ClassTag](f: T => B): Tensor1D[B] =  Tensor1D[B](data.map(f), shape)
  override def toString: String = "Tensor1D(" + data.mkString(", ") + ")"
  override def toList: List[T] = data.toList

  override def apply(idx: Int*): T = data(idx.head)
  override def at(idx: Seq[Int]): T = data(idx.head)
}

object Tensor1D {
  def fromRangeTree[T: ClassTag](a: Array[T], rg: RangeTree): Tensor1D[T] = {
    require(RangeTree.depth(rg) == 1, s"Cannot contstruct Tensor1D from tree $rg")
    val shape = RangeTree.shape(rg)

    val data: Array[T] =
      rg match {
        case Leaf(begin, end) => a.slice(begin, end)
        case b: Branch => throw new Exception("Malformed RangeTree for 1D tensor")
      }

    new Tensor1D(data, shape)
  }
}

case class Tensor2D[T](data: Array[Tensor1D[T]], shape: List[Int]) extends TensorStatic[T] {
  override def rank: Int = 2
  override def length: Int = data.length
  override def size: Int = data.map(_.size).sum

  override def map[B: ClassTag](f: T => B): Tensor2D[B] =
    Tensor2D[B](data.map(_.map(f).asInstanceOf[Tensor1D[B]]), shape)
  override def toString: String = "Tensor2D(" + data.mkString(", ") + ")"
  override def toList: List[List[T]] =
    data.collect { case t1: Tensor1D[T] => t1.toList }.toList

  override def apply(idx: Int*): T = data(idx.head).at(idx.tail)
  override def at(idx: Seq[Int]): T = data(idx.head).at(idx.tail)
}

object Tensor2D {
  def fromRangeTree[T: ClassTag](a: Array[T], rg: RangeTree): Tensor2D[T] = {
    require(RangeTree.depth(rg) == 2, s"Cannot contstruct Tensor2D from tree $rg")
    val shape = RangeTree.shape(rg)

    val data: Array[Tensor1D[T]] =
      rg match {
        case Branch(children) => children.collect{
          case l: Leaf => Tensor1D.fromRangeTree(a, l)
          case b: Branch => throw new Exception("Malformed RangeTree for 2D tensor")
        }.toArray
        case l: Leaf => throw new Exception("Malformed RangeTree for 2D tensor")
      }

    new Tensor2D(data, shape)
  }
}



case class Tensor3D[T](data: Array[Tensor2D[T]], shape: List[Int]) extends TensorStatic[T] {
  override def rank: Int = 3
  override def length: Int = data.length
  override def size: Int = data.map(_.size).sum

  override def map[B: ClassTag](f: T => B): Tensor3D[B] =
    Tensor3D[B](data.map(_.map(f).asInstanceOf[Tensor2D[B]]), shape)
  override def toString: String = "Tensor3D(" + data.mkString(", ") + ")"
  override def toList: List[List[List[T]]] = data.collect { _.toList }.toList

  override def apply(idx: Int*): T = data(idx.head).at(idx.tail)
  override def at(idx: Seq[Int]): T = data(idx.head).at(idx.tail)

}

object Tensor3D {
  def fromRangeTree[T: ClassTag](a: Array[T], rg: RangeTree): Tensor3D[T] = {
    require(RangeTree.depth(rg) == 3, s"Cannot contstruct Tensor3D from tree $rg")
    val shape = RangeTree.shape(rg)

    val data: Array[Tensor2D[T]] =
      rg match {
        case Branch(children) => children.collect{
          case b: Branch => Tensor2D.fromRangeTree(a, b)
          case l: Leaf  => throw new Exception("Malformed RangeTree for 2D tensor")
        }.toArray
        case l: Leaf => throw new Exception("Malformed RangeTree for 2D tensor")
      }

    new Tensor3D(data, shape)
  }
}
