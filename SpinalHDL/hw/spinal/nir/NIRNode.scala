// src/main/scala/nir/NIRNode.scala
package nir

sealed trait NIRNode {
  def id: String
  def previous: Set[String]
}

case class LIF(
  id: String,
  tau: Array[Float],
  r: Array[Float],
  v_leak: Array[Float],
  v_threshold: Array[Float],
  previous: Set[String]
) extends NIRNode

case class LI(
  id: String,
  tau: Array[Float],
  r: Array[Float],
  v_leak: Array[Float],
  previous: Set[String]
) extends NIRNode

case class CubaLIF(
  id: String,
  tau: Array[Float],
  tauSynExc: Array[Float],
  tauSynInh: Array[Float],
  previous: Set[String]
) extends NIRNode

case class Conv2d(
  id: String,
  kernelSize: (Int,Int),
  numFilters: Int,
  stride: (Int,Int),
  padding: (Int,Int),
  bias: Array[Array[Float]],
  weight: Array[Array[Float]],
  previous: Set[String]
) extends NIRNode

case class Linear(
  id: String,
  weight: Array[Float],
  previous: Set[String]
) extends NIRNode


case class Affine(
  id: String,
  bias: Array[Float],
  weight: Array[Array[Float]],
  previous: Set[String]
) extends NIRNode


case class Input(
  id: String,
  shape: Array[Long],
  previous: Set[String]


) extends NIRNode

case class Output(
  id: String,
  shape: Array[Long],
  previous: Set[String]

) extends NIRNode
