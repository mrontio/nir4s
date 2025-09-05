package nir.tensor

import cats.data.State
import cats.syntax.traverse._
import cats.instances.list._
import shapeless.HList._


sealed trait Shape

// Tree of indices for flat array.
sealed trait RangeTree
case class Leaf(begin: Int, end: Int) extends RangeTree
case class Branch(children: List[RangeTree]) extends RangeTree


object RangeTree {
  // Counter monad
  private type Counter[A] = State[Int, A]
  private val next: Counter[Int] = State { s => (s + 1, s) }
  private def plus(x: Int): Counter[(Int, Int)] =
    State(s => {
      val b = s
      val e = s + x
      (e, (b, e))
    })

  // Build the range tree
  def buildTree(shape: List[Int]): RangeTree = {
    def build(dims: List[Int]): Counter[RangeTree] =
      dims match {
        case Nil =>
          // Leaf dimension size 1
          for { idx <- next } yield Leaf(idx, idx + 1)
        case dim :: Nil =>
          // Leaf dimension size n > 1
          for { be <- plus(dim) } yield Leaf(be._1, be._2)
        case h :: t =>
          for { children <- List.fill(h)(build(t)).sequence } yield Branch(children)
      }
    build(shape).runA(0).value
  }
}
