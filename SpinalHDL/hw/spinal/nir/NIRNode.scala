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
)

sealed trait NIRParams

final case class LIFParams(
  tau: Array[Float],
  r: Array[Float],
  v_leak: Array[Float],
  v_threshold: Array[Float],
) extends NIRParams

final case class LIParams(
  tau: Array[Float],
  r: Array[Float],
  v_leak: Array[Float],
) extends NIRParams

final case class CubaLIFParams(
  tau: Array[Float],
  tauSynExc: Array[Float],
  tauSynInh: Array[Float],
) extends NIRParams

final case class Conv2dParams(
  kernelSize: (Int,Int),
  numFilters: Int,
  stride: (Int,Int),
  padding: (Int,Int),
  bias: Array[Array[Float]],
  weight: Array[Array[Float]],
) extends NIRParams

final case class LinearParams(
  weight: Array[Float],
) extends NIRParams

final case class AffineParams(
  bias: Array[Float],
  weight: Array[Array[Float]],
) extends NIRParams


final case class InputParams(
  shape: Array[Long],
) extends NIRParams

final case class OutputParams(
  shape: Array[Long],
) extends NIRParams
