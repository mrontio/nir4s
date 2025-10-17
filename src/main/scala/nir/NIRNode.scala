// src/main/scala/nir/NIRNode.scala
package nir

import tensor._

/** Unresolved graph node identified only by string IDs for its
  * predecessors.
  *
  * @param id unique identifier for this node
  * @param prevIds identifiers of predecessor nodes
  * @param params operation parameters for this node
  */
case class RawNode(
  id:      String,
  prevIds: Set[String],
  params:  NIRParams
)

/** Resolved node within a NIR graph.
  *
  * @param id unique identifier for this node
  * @param previous actual predecessor nodes
  * @param params operation parameters associated with this node
  */
case class NIRNode(
  id:       String,
  var previous: Set[NIRNode], // TODO: This should not be var, I need this temporarily.
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

/** Parameters describing a one-dimensional convolution operation.
  *
  * @param weight filter weights with shape `(out, in, k)`
  * @param bias bias tensor
  * @param stride stride along the spatial dimension
  * @param padding zero-padding applied to the input
  * @param dilation dilation factor of the kernel
  * @param groups number of groups the input is split into
  * @param input_shape length of the input signal
  */
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

/** Parameters describing a two-dimensional convolution operation.
  *
  * @param weight filter weights `(out, in, kH, kW)`
  * @param bias bias tensor
  * @param stride strides along height and width
  * @param padding padding along height and width
  * @param dilation dilation factors
  * @param groups grouping factor
  * @param input_shape spatial dimensions of the input
  */
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


/** A temporary class that is not part of the NIR standard but I need for our research.
  * To be removed once subgraphing functionality has been implemented
  * This is a subgraph Conv2D -> IF
  */
final case class Conv2DIFParams(
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

  override def nirType: String = "Conv2dIF"
  override def toString: String = {
    val weightString =  s"\n\t\tWeight shape: ${weight.shape},\n\t\tKernel size: ${kernelSize}\n\t\tChannels in: ${inChannels},\n\t\tChannels out: ${outChannels}"
    s"$nirType {\n\tweights = $weightString,\n\tbias = ${bias.shape},\n\tstride = $stride,\n\tpadding = $padding,\n\tdilation = $dilation,\n\tgroups = $groups,\n\tinput_shape = $input_shape\n}"
  }
}



/** Parameters for flattening a tensor into a lower rank.
  *
  * @param start_dim first dimension to flatten
  * @param end_dim last dimension to flatten
  * @param input_type original tensor shape
  */
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

/** Parameters for two-dimensional sum pooling.
  *
  * @param kernel_size size of the pooling window
  * @param padding padding applied before pooling
  * @param stride step of the pooling window
  */
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


/** Leaky integrator neuron parameters.
  *
  * @param tau time constant
  * @param r resistance
  * @param v_leak leak potential
  */
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

/** Integrate-and-fire neuron parameters.
  *
  * @param r membrane resistance
  * @param v_reset reset potential
  * @param v_threshold firing threshold
  */
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

/** Leaky integrate-and-fire neuron parameters.
  *
  * @param tau membrane time constant
  * @param r membrane resistance
  * @param v_leak leak potential
  * @param v_threshold firing threshold
  */
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

/** Current-based LIF neuron parameters with excitatory and inhibitory time constants.
  *
  * @param tau membrane time constant
  * @param tauSynExc excitatory synapse constant
  * @param tauSynInh inhibitory synapse constant
  */
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

/** Parameters of a linear transformation.
  *
  * @param weight transformation matrix
  */
final case class LinearParams(
  weight: TensorStatic[Float],
) extends NIRParams {
  override def nirType: String = "Linear"
  override def toString: String = {
    s"$nirType {\n\tweight = ${weight.shape}\n}"
  }
}

/** Parameters of an affine transformation.
  *
  * @param bias bias vector
  * @param weight weight matrix
  */
final case class AffineParams(
  bias: TensorStatic[Float],
  weight: TensorStatic[Float],
) extends NIRParams {
  override def nirType: String = "Affine"
  override def toString: String = {
    s"$nirType {\n\tbias = ${bias.shape},\n\tweight = ${weight.shape}\n}"
  }
}

/** Parameters describing the input node of a graph.
  *
  * @param shape shape of the incoming tensor
  */
final case class InputParams(
  shape: TensorStatic[Long],
) extends NIRParams {
  override def nirType: String = "Input"
  override def toString: String = {
    s"$nirType {\n\tshape = $shape\n}"
  }
}

/** Parameters describing the output node of a graph.
  *
  * @param shape shape of the produced tensor
  */
final case class OutputParams(
  shape: TensorStatic[Long],
) extends NIRParams {
  override def nirType: String = "Output"
  override def toString: String = {
    s"$nirType {\n\tshape = $shape\n}"
  }
}
