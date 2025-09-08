// src/main/scala/nir/NIRNode.scala
package nir

import tensor._

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
    s"$id: previous=$prevIds, params=$params)"
  }
}

sealed trait NIRParams {
  def nirType: String
  def toString: String
}

final case class Conv1DParams(
  weight: TensorStatic[Float],
  bias: TensorStatic[Float],
  stride: TensorStatic[Long],
  padding: TensorStatic[Long],
  dilation: TensorStatic[Long],
  groups: Long,
  input_shape: Long,
) extends NIRParams {
  def outChannels: List[Int] =
    List(weight.shape(0))

  def inChannels: List[Int] =
    List(weight.shape(1))

  def kernelSize: List[Int] =
    List(weight.shape(2))

  override def nirType: String = "Conv1d"
  override def toString: String = {
    val weightString = s"\n\t\tWeight shape: ${weight.shape},\n\t\tKernel size: ${kernelSize}\n\t\tChannels in: ${inChannels},\n\t\tChannels out: ${outChannels}"
    s"Conv1D {\n\tweight = $weightString,\n\tbias = ${bias.shape},\n\tstride = $stride,\n\tpadding = $padding,\n\tdilation = $dilation,\n\tgroups = $groups,\n\tinput_shape = $input_shape\n}"
  }
}

final case class Conv2DParams(
  weight: TensorStatic[Float],
  bias: TensorStatic[Float],
  stride: TensorStatic[Long],
  padding: TensorStatic[Long],
  dilation: TensorStatic[Long],
  groups: Long,
  input_shape: TensorStatic[Long]
) extends NIRParams {
    def outChannels: List[Int] =
    List(weight.shape(0))

  def inChannels: List[Int] =
    List(weight.shape(1))

  def kernelSize: List[Int] =
    List(weight.shape(2), weight.shape(3))

  override def nirType: String = "Conv2d"
  override def toString: String = {
    val weightString =  s"\n\t\tWeight shape: ${weight.shape},\n\t\tKernel size: ${kernelSize}\n\t\tChannels in: ${inChannels},\n\t\tChannels out: ${outChannels}"
    s"$nirType {\n\tweights = $weightString,\n\tbias = ${bias.shape},\n\tstride = $stride,\n\tpadding = $padding,\n\tdilation = $dilation,\n\tgroups = $groups,\n\tinput_shape = $input_shape\n}"
  }
}

final case class FlattenParams(
  start_dim: Long,
  end_dim: Long,
  input_type: TensorStatic[Long]
) extends NIRParams {
  override def nirType: String = "Flatten"
  override def toString: String = {
    s"$nirType {\n\tstart_dim = $start_dim,\n\tend_dim = $end_dim,\n\tinput_type = $input_type\n}"
  }
}

final case class SumPool2DParams(
  kernel_size: TensorStatic[Long],
  padding: TensorStatic[Long],
  stride: TensorStatic[Long]
) extends NIRParams {
  override def nirType: String = "SumPool2d"
  override def toString: String = {
    s"$nirType {\n\tkernel = $kernel_size,\n\tpadding = $padding,\n\tstride = $stride\n}"
  }
}


final case class LIParams(
  tau: TensorStatic[Float],
  r: TensorStatic[Float],
  v_leak: TensorStatic[Float],
) extends NIRParams {
  override def nirType: String = "LI"
  override def toString: String = {
    s"$nirType {\n\ttau = ${tau.shape},\n\tr = ${r.shape},\n\tv_leak = ${v_leak.shape}\n}"
  }
}

final case class IFParams(
  r: TensorStatic[Float],
  v_reset: TensorStatic[Float],
  v_threshold: TensorStatic[Float],
) extends NIRParams {
  override def nirType: String = "IF"
  override def toString: String = {
    s"$nirType {\n\tr = ${r.shape},\n\tv_reset = ${v_reset.shape},\n\tv_threshold = ${v_threshold.shape}\n}"
  }
}

final case class LIFParams(
  tau: TensorStatic[Float],
  r: TensorStatic[Float],
  v_leak: TensorStatic[Float],
  v_threshold: TensorStatic[Float],
) extends NIRParams {
  override def nirType: String = "LIF"
  override def toString: String = {
    s"$nirType {\n\ttau = ${tau.shape},\n\tr = ${r.shape},\n\tv_leak = ${v_leak.shape},\n\tv_threshold = ${v_threshold.shape}\n}"
  }
}

final case class CubaLIFParams(
  tau: TensorStatic[Float],
  tauSynExc: TensorStatic[Float],
  tauSynInh: TensorStatic[Float],
) extends NIRParams {
  override def nirType: String = "CubaLIF"
  override def toString: String = {
    s"$nirType {\n\ttau = ${tau.shape},\n\ttauSynExc = ${tauSynExc.shape},\n\ttauSynInh = ${tauSynInh.shape}\n}"
  }
}

final case class LinearParams(
  weight: TensorStatic[Float],
) extends NIRParams {
  override def nirType: String = "Linear"
  override def toString: String = {
    s"$nirType {\n\tweight = ${weight.shape}\n}"
  }
}

final case class AffineParams(
  bias: TensorStatic[Float],
  weight: TensorStatic[Float],
) extends NIRParams {
  override def nirType: String = "Affine"
  override def toString: String = {
    s"$nirType {\n\tbias = ${bias.shape},\n\tweight = ${weight.shape}\n}"
  }
}

final case class InputParams(
  shape: TensorStatic[Long],
) extends NIRParams {
  override def nirType: String = "Input"
  override def toString: String = {
    s"$nirType {\n\tshape = $shape\n}"
  }
}

final case class OutputParams(
  shape: TensorStatic[Long],
) extends NIRParams {
  override def nirType: String = "Output"
  override def toString: String = {
    s"$nirType {\n\tshape = $shape\n}"
  }
}
