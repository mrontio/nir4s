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
