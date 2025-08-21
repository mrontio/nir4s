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

sealed trait NIRParams {
  def toString: String
}

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
) extends NIRParams {
  override def toString: String = {
    s"Conv1D {\n\tweights = $weights,\n\tbias = $bias,\n\tstride = $stride,\n\tpadding = $padding,\n\tdilation = $dilation,\n\tgroups = $groups,\n\tinput_shape = $input_shape\n}"
  }
}
final case class FlattenParams(
  start_dim: Long,
  end_dim: Long,
  input_type: Tensor[Long]
) extends NIRParams {
  override def toString: String = {
    s"FlattenParams {\n\tstart_dim = $start_dim,\n\tend_dim = $end_dim,\n\tinput_type = $input_type\n}"
  }
}

final case class LIParams(
  tau: Tensor[Float],
  r: Tensor[Float],
  v_leak: Tensor[Float],
) extends NIRParams {
  override def toString: String = {
    s"LIParams {\n\ttau = $tau,\n\tr = $r,\n\tv_leak = $v_leak\n}"
  }
}

final case class IFParams(
  r: Tensor[Float],
  v_reset: Tensor[Float],
  v_threshold: Tensor[Float],
) extends NIRParams {
  override def toString: String = {
    s"IFParams {\n\tr = $r,\n\tv_reset = $v_reset,\n\tv_threshold = $v_threshold\n}"
  }
}

final case class LIFParams(
  tau: Tensor[Float],
  r: Tensor[Float],
  v_leak: Tensor[Float],
  v_threshold: Tensor[Float],
) extends NIRParams {
  override def toString: String = {
    s"LIFParams {\n\ttau = $tau,\n\tr = $r,\n\tv_leak = $v_leak,\n\tv_threshold = $v_threshold\n}"
  }
}

final case class CubaLIFParams(
  tau: Tensor[Float],
  tauSynExc: Tensor[Float],
  tauSynInh: Tensor[Float],
) extends NIRParams {
  override def toString: String = {
    s"CubaLIFParams {\n\ttau = $tau,\n\ttauSynExc = $tauSynExc,\n\ttauSynInh = $tauSynInh\n}"
  }
}

final case class LinearParams(
  weight: Tensor1D[Float],
) extends NIRParams {
  override def toString: String = {
    s"LinearParams {\n\tweight = $weight\n}"
  }
}

final case class AffineParams(
  bias: Tensor[Float],
  weight: Tensor[Float],
) extends NIRParams {
  override def toString: String = {
    s"AffineParams {\n\tbias = $bias,\n\tweight = $weight\n}"
  }
}

final case class InputParams(
  shape: Tensor1D[Long],
) extends NIRParams {
  override def toString: String = {
    s"InputParams {\n\tshape = $shape\n}"
  }
}

final case class OutputParams(
  shape: Tensor1D[Long],
) extends NIRParams {
  override def toString: String = {
    s"OutputParams {\n\tshape = $shape\n}"
  }
}
