package nir

import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag

// Base trait / abstract class
sealed trait Tensor[T] {
  def shape: Seq[Int]
  def rank: Int = shape.length

  protected def squeezableIndices: Set[Int] = {
    shape.zipWithIndex.collect { case (s, i) if s <= 1 => i }.toSet
  }
  def squeezeable: Boolean = squeezableIndices.size > 0
  def map[B: ClassTag](f: T => B): Tensor[B]
}

case class Tensor1D[T](data: Array[T]) extends Tensor[T] {
  override def shape: Seq[Int] = Seq(data.length)
  override def map[B: ClassTag](f: T => B): Tensor[B] = {
     Tensor1D[B](data.map(f))
  }
}

object Tensor1D {
  def apply[T: ClassTag](values: T*): Tensor1D[T] = {
    new Tensor1D[T](values.toArray)
  }
}

case class Tensor2D[T](data: Array[Tensor1D[T]]) extends Tensor[T] {
  override def shape: Seq[Int] =
    Seq(data.length) ++ (data.headOption match {
      case Some(t: Tensor1D[T]) => t.shape
      case None => Seq(0)
    })

    override def map[B: ClassTag](f: T => B): Tensor2D[B] = {
      Tensor2D[B](data.map(_.map(f).asInstanceOf[Tensor1D[B]]))
    }
}

object Tensor2D {
  def apply[T](values: Tensor1D[T]*): Tensor2D[T] = {
    new Tensor2D(values.toArray)
  }

  def apply[T](array: Array[Tensor1D[T]]) = new Tensor2D[T](array)
}

case class Tensor3D[T](data: Array[Tensor2D[T]]) extends Tensor[T] {
  override def shape: Seq[Int] =
    Seq(data.length) ++ (data.headOption match {
      case Some(t: Tensor2D[T]) => t.shape
      case None => Seq(0, 0)
    })

    override def map[B: ClassTag](f: T => B): Tensor3D[B] = {
      Tensor3D[B](data.map(_.map(f).asInstanceOf[Tensor2D[B]]))
    }
}

object Tensor3D {
  def apply[T](values: Tensor2D[T]*): Tensor3D[T] = {
    new Tensor3D(values.toArray)
  }

  def apply[T](array: Array[Tensor2D[T]]) = new Tensor3D[T](array)
}

case class Tensor4D[T](data: Array[Tensor3D[T]]) extends Tensor[T] {
  override def shape: Seq[Int] =
    Seq(data.length) ++ (data.headOption match {
      case Some(t: Tensor3D[T]) => t.shape
      case None => Seq(0, 0, 0)
    })

    def apply[T](array: Array[Tensor3D[T]]) = new Tensor4D[T](array)
}

object Tensor4D {
  def apply[T](values: Tensor3D[T]*): Tensor4D[T] =  new Tensor4D(values.toArray)
  def apply[T](array: Array[Tensor3D[T]]) = new Tensor4D[T](array)
}

object Tensor {
  // We want to give the map because we don't want to lose type info
  def fromHDFMap[T](key: String, hdfMap: Map[String, Any])(implicit ev: reflect.ClassTag[T]): Tensor[T] = {
    val attr = hdfMap(key)
    val t: Tensor[T] = attr.getClass.getName match {
      case "[F" if ev == reflect.classTag[Float] => fromArray1D(attr.asInstanceOf[Array[T]])
      case "[[F" if ev == reflect.classTag[Float] => fromArray2D(attr.asInstanceOf[Array[Array[T]]])
      case "[[[F" if ev == reflect.classTag[Float] => fromArray3D(attr.asInstanceOf[Array[Array[Array[T]]]])
      case "[[[[F" if ev == reflect.classTag[Float] => fromArray4D(attr.asInstanceOf[Array[Array[Array[Array[T]]]]])
      case a => throw new java.text.ParseException(s"Expected to read float tensor but read \"${a}\"", 0)
    }

    // Get rid of singleton dimensions
    squeeze(t)
    //t
  }

  private def fromArray1D[T](a: Array[T]): Tensor1D[T] = Tensor1D(a)
  private def fromArray2D[T](a: Array[Array[T]]): Tensor2D[T] =  Tensor2D(a.map(fromArray1D[T](_)))
  private def fromArray3D[T](a: Array[Array[Array[T]]]): Tensor3D[T] =  Tensor3D(a.map(fromArray2D[T](_)))
  private def fromArray4D[T](a: Array[Array[Array[Array[T]]]]): Tensor4D[T] =  Tensor4D(a.map(fromArray3D[T](_)))

  // This is a bad function with known issues:
  // - Not generalized, as you can see I am hand-crafting each index here
  // - Error prone, there is no exception checking
  // - We assume we _only remove one index_
  // In the future, look at the Pytorch implementation

  def squeeze[T: ClassTag](t: Tensor[T]): Tensor[T] = {
    t match {
      case t1: Tensor1D[T] => t1
      case t2: Tensor2D[T] => {
        // Squeeze from the left towards index zero
        val ss = t2.squeezableIndices - 0
        var result: Array[T] = Array.empty[T]
        for (s <- ss) {
          s match {
            case 1 => result = t2.data.collect {
              case t1: Tensor1D[T] => t1.data
            }.flatten
          }
        }
        Tensor1D(result)
      }
      case x => throw new NotImplementedError(s"squeezing array of dimensions $x is not yet supported")
     }
  }
}
