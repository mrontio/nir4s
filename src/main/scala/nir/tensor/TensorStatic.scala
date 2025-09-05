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
  def shape: List[Int]
  def rank: Int
  def size: Int

  def toString: String

  def map[B: ClassTag](f: T => B): TensorStatic[B]

  // Ambiguous type will be over-written by children
  def toList: List[_]
}

case class Tensor1D[T](data: Array[T], shape: List[Int]) extends TensorStatic[T] {
  // TODO: Somehow make this compile-safe, _without_ writing 500 lines of code.
  require(shape.length == 1)

  override def rank: Int = 1
  override def size: Int = data.length

  override def map[B: ClassTag](f: T => B): TensorStatic[B] =  Tensor1D[B](data.map(f), shape)
  override def toString: String = "Tensor1D(" + data.mkString(", ") + ")"
  override def toList: List[T] = data.toList
  def apply(i0: Int): T = data(i0)

}


case class Tensor2D[T](data: Array[Tensor1D[T]], shape: List[Int]) extends TensorStatic[T] {
  override def rank: Int = 2
  override def size: Int = data.map(_.size).sum

  def map[B: ClassTag](f: T => B): Tensor2D[B] =
    Tensor2D[B](data.map(_.map(f).asInstanceOf[Tensor1D[B]]), shape)
  override def toString: String = "Tensor2D(" + data.mkString(", ") + ")"
  override def toList: List[List[T]] =
    data.collect { case t1: Tensor1D[T] => t1.toList }.toList
  def apply(i0: Int, i1: Int): T = data(i0)(i1)
}

object Tensor2D {
  def fromRangeTree[T: ClassTag](a: Array[T], rg: RangeTree): Tensor2D[T] = {
    require(RangeTree.depth(rg) == 2, s"Cannot contstruct Tensor2D from tree $rg")
    val shape = RangeTree.shape(rg)

    val data: Array[Tensor1D[T]] =
      rg match {
        case Branch(children) => children.collect{
          case Leaf(begin, end) => Tensor1D(a.slice(begin, end), List(begin - end))
          case b: Branch => throw new Exception("Malformed RangeTree for 2D tensor")
        }.toArray
        case l: Leaf => throw new Exception("Malformed RangeTree for 2D tensor")
      }

    new Tensor2D(data, shape)
  }
}
