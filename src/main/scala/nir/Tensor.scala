package nir

import io.jhdf.api.{Dataset}
import scala.reflect.ClassTag

sealed trait RangeTree
case class Leaf(begin: Int, end: Int) extends RangeTree
case class Branch(children: List[RangeTree]) extends RangeTree


class Counter(var value: Int) {
  def next(): Int = {
    val current = value
    value += 1
    current
  }

  def plus(x: Int): (Int, Int) = {
    val current = value
    val plussed = current + x
    value = plussed
    (current, plussed)
  }
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


class Tensor[D](data: Array[D], idx: Indexer) {
  val shape = idx.shape
  val rank = idx.rank
  val size = idx.size
  def apply(indices: Int*) = data(idx(indices))
  def reshape(newshape: List[Int]): Tensor[D] = new Tensor[D](data, idx.reshape(newshape))
  def map[B: ClassTag](f: D => B): Tensor[B] = new Tensor[B](data.map(f), idx)

  private def toNestedList(tree: RangeTree): List[_] = tree match {
    case Leaf(b, e) =>
      data.slice(b, e).toList

    case Branch(children) =>
      children.map(toNestedList)
  }

  def toList: List[_] = toNestedList(idx.rangeTree)
}

object Tensor {
  def apply[D](a: Array[D], shape: List[Int]): Tensor[D] = {
    val i = Indexer(shape)
    if (i.size != a.length) throw new IllegalArgumentException(s"Supplied array is size ${a.length} but shape is over size ${i.size}.")
    new Tensor[D](a, i)

  }

  // Allow to match based on rank
  object Rank {
    def unapply[D](tensor: Tensor[D]): Option[Int] = Some(tensor.rank)
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


  def apply(d: Dataset): Tensor[_] = {
    val idx = Indexer(d.getDimensions.toList)
    d.getDataType.getJavaType.toString match {
      case "float" => new Tensor(flattenDatasetArray[Float](d), idx)
      case "int"   => new Tensor(flattenDatasetArray[Int](d), idx)
      case "long"  => new Tensor(flattenDatasetArray[Long](d), idx)
      case o => throw new java.text.ParseException(s"Java type \"${o}\" not yet supported for Tensor conversion", 0)
    }

  }
}
