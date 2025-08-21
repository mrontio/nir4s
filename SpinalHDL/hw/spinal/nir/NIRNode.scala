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
  get: Tensor3D[Float]
) extends ConvWeights[Tensor3D[Float]] {
  def outChannels: Tensor3D[Float] => Set[Int] =
    w => Set(w.shape(0))

  def inChannels: Tensor3D[Float] => Set[Int] =
    w => Set(w.shape(1))

  def kernelSize: Tensor3D[Float] => Set[Int] =
    w => Set(w.shape(2))
}

final case class Conv1DParams(
  weights: Conv1DWeights,
  bias: Tensor1D[Float],
  stride: Tensor1D[Long],
  padding: Tensor1D[Long],
  dilation: Tensor1D[Long],
  groups: Long,
  input_shape: Long
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

final case class FlattenParams(
  start_dim: Long,
  end_dim: Long,
  input_type: Tensor[Long]
) extends NIRParams

final case class LIParams(
  tau: Tensor[Float],
  r: Tensor[Float],
  v_leak: Tensor[Float],
) extends NIRParams

final case class IFParams(
  r: Tensor[Float],
  v_reset: Tensor[Float],
  v_threshold: Tensor[Float],
) extends NIRParams

final case class LIFParams(
  tau: Tensor[Float],
  r: Tensor[Float],
  v_leak: Tensor[Float],
  v_threshold: Tensor[Float],
) extends NIRParams

final case class CubaLIFParams(
  tau: Tensor[Float],
  tauSynExc: Tensor[Float],
  tauSynInh: Tensor[Float],
) extends NIRParams

final case class LinearParams(
  weight: Tensor1D[Float],
) extends NIRParams

final case class AffineParams(
  bias: Tensor[Float],
  weight: Tensor[Float],
) extends NIRParams


final case class InputParams(
  shape: Tensor1D[Long],
) extends NIRParams

final case class OutputParams(
  shape: Tensor1D[Long],
) extends NIRParams
