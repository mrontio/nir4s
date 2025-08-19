// src/main/scala/nir/NIRNode.scala
package nir

case class RawNode(
  id:      String,
  prevIds: Set[String],
  params:  NIRParams
)

case class NIRNode(
  id:       String,
  previous: Set[NIRNode],
  params:   NIRParams
) {
  override def toString: String = {
    val prevIds = previous.map(_.id).mkString("{", ", ", "}")
    s"NIRNode(id=$id, previous=$prevIds, params=$params)"
  }
}

sealed trait NIRParams
sealed trait ConvWeights[W] {
  def get: W
  def outChannels: W => Set[Int]
  def inChannels: W => Set[Int]
  def kernelSize: W => Set[Int]
}

final case class Conv1DWeights(
  get: nir.Matrix3D[Float]
) extends ConvWeights[nir.Matrix3D[Float]] {
  def outChannels: nir.Matrix3D[Float] => Set[Int] =
    w => Set(w.length)

  def inChannels: nir.Matrix3D[Float] => Set[Int] =
    w => if (w.nonEmpty) Set(w.head.length) else Set.empty

  def kernelSize: nir.Matrix3D[Float] => Set[Int] =
    w => if (w.nonEmpty && w.head.nonEmpty) Set(w.head.head.length) else Set.empty
}

final case class Conv1DParams(
  weights: Conv1DWeights,
  bias: nir.Matrix1D[Float],
  stride: Array[Long],
  padding: Array[Long],
  dilation: Array[Long],
  groups: Long,
  input_shape: Long
) extends NIRParams

final case class LIFParams(
  tau: Tensor[Float],
  r: Tensor[Float],
  v_leak: Tensor[Float],
  v_threshold: Tensor[Float],
) extends NIRParams

final case class LIParams(
  tau: Tensor[Float],
  r: Tensor[Float],
  v_leak: Tensor[Float],
) extends NIRParams

final case class CubaLIFParams(
  tau: Tensor[Float],
  tauSynExc: Tensor[Float],
  tauSynInh: Tensor[Float],
) extends NIRParams

final case class Conv2dParams(
  dilation: (Float, Float),
  groups: Int,
  input_shape: (Int, Int),
  padding: (Int,Int),
  stride: (Int,Int),
  bias: nir.Matrix1D[Float],
  weight: nir.Matrix4D[Float],
  kernelSize: (Int,Int), // Computed from weight
) extends NIRParams

final case class LinearParams(
  weight: Tensor[Float],
) extends NIRParams

final case class AffineParams(
  bias: Tensor[Float],
  weight: Tensor[Float],
) extends NIRParams


final case class InputParams(
  shape: Array[Int],
) extends NIRParams

final case class OutputParams(
  shape: Array[Int],
) extends NIRParams
