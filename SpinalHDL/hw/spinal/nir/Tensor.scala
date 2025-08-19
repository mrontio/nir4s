package nir

import scala.collection.immutable.ArraySeq

// Base trait / abstract class
sealed trait Tensor {
  def shape: Seq[Int]
  def rank: Int = shape.length

  protected def squeezableIndices: Set[Int] = {
    shape.zipWithIndex.collect { case (s, i) if s <= 1 => i }.toSet
  }
  def squeezeable: Boolean = squeezableIndices.size > 0
}

case class Tensor1D(data: Array[Float]) extends Tensor {
    override def shape: Seq[Int] = Seq(data.length)
}

case class Tensor2D(data: Array[Tensor1D]) extends Tensor {
  override def shape: Seq[Int] =
    Seq(data.length) ++ (data.headOption match {
      case Some(t: Tensor1D) => t.shape
      case None => Seq(0)
    })
}

case class Tensor3D(data: Array[Tensor2D]) extends Tensor {
  override def shape: Seq[Int] =
    Seq(data.length) ++ (data.headOption match {
      case Some(t: Tensor2D) => t.shape
      case None => Seq(0, 0)
    })
}

case class Tensor4D(data: Array[Tensor3D]) extends Tensor {
  override def shape: Seq[Int] =
    Seq(data.length) ++ (data.headOption match {
      case Some(t: Tensor3D) => t.shape
      case None => Seq(0, 0, 0)
    })
}


object Tensor {
  // We want to give the map because we don't want to lose type info
  def fromHDFMap(key: String, hdfMap: Map[String, Any]): Tensor = {
    val attr = hdfMap(key)
    val t: Tensor = attr.getClass.getName match {
      case "[F" => fromArray1D(attr.asInstanceOf[Array[Float]])
      case "[[F" => fromArray2D(attr.asInstanceOf[Array[Array[Float]]])
      case "[[[F" => fromArray3D(attr.asInstanceOf[Array[Array[Array[Float]]]])
      case "[[[[F" => fromArray4D(attr.asInstanceOf[Array[Array[Array[Array[Float]]]]])
      case a => throw new java.text.ParseException(s"Expected to read float tensor but read \"${a}\"", 0)
    }

    // Get rid of singleton dimensions
    squeeze(t)
    //t
  }

  private def fromArray1D(a: Array[Float]): Tensor1D = Tensor1D(a)
  private def fromArray2D(a: Array[Array[Float]]): Tensor2D =  Tensor2D(a.map(fromArray1D(_)))
  private def fromArray3D(a: Array[Array[Array[Float]]]): Tensor3D =  Tensor3D(a.map(fromArray2D(_)))
  private def fromArray4D(a: Array[Array[Array[Array[Float]]]]): Tensor4D =  Tensor4D(a.map(fromArray3D(_)))

  // This is a bad function with known issues:
  // - Not generalized, as you can see I am hand-crafting each index here
  // - Error prone, there is no exception checking
  // - We assume we _only remove one index_
  // In the future, look at the Pytorch implementation

  def squeeze(t: Tensor): Tensor = {
    t match {
      case t1: Tensor1D => t1
      case t2: Tensor2D => {
        // Squeeze from the left towards index zero
        val ss = t2.squeezableIndices - 0
        var result: Array[Float] = Array.empty[Float]
        for (s <- ss) {
          s match {
            case 1 => result = t2.data.collect {
              case t1: Tensor1D => t1.data
            }.flatten
          }
        }
        Tensor1D(result)
      }
      case x => throw new NotImplementedError(s"squeezing array of dimensions $x is not yet supported")
     }
  }
}
