package nir

// Base trait / abstract class
sealed trait Tensor {
  def shape: Seq[Int]
  def rank: Int = shape.length
}

// Specializations
final case class Tensor1D(values: Array[Float]) extends Tensor {
  override def shape: Seq[Int] = Seq(values.length)
}

final case class Tensor2D(values: Array[Array[Float]]) extends Tensor {
  override def shape: Seq[Int] = Seq(values.length, values.headOption.map(_.length).getOrElse(0))
}

final case class Tensor3D(values: Array[Array[Array[Float]]]) extends Tensor {
  override def shape: Seq[Int] =
    Seq(values.length,
        values.headOption.map(_.length).getOrElse(0),
        values.headOption.flatMap(_.headOption).map(_.length).getOrElse(0))
}

final case class Tensor4D(values: Array[Array[Array[Array[Float]]]]) extends Tensor {
  override def shape: Seq[Int] =
    Seq(values.length,
        values.headOption.map(_.length).getOrElse(0),
        values.headOption.flatMap(_.headOption).map(_.length).getOrElse(0),
        values.headOption.flatMap(_.headOption).flatMap(_.headOption).map(_.length).getOrElse(0))
}
