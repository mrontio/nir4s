package nir

import munit.FunSuite
import java.io.File
import java.nio.file.{Files, Paths}
import io.jhdf.HdfFile
import io.jhdf.api.{Attribute, Dataset, Group, Node}
import scala.jdk.CollectionConverters._

import io.circe.parser._

import tensor._

class TensorSpec extends FunSuite {
  val conv1Path = "src/test/scala/nir/samples/conv1/network.nir"
  val file = new File(conv1Path)
  val hdf = new HdfFile(file)
  val nodeHDF = hdf.getByPath("/node/nodes")
  val d: Dataset = nodeHDF match {
    case g1: Group => g1.getChild("0") match {
      case g2: Group => g2.getChild("weight").asInstanceOf[Dataset]
    }
  }

  test("Tensor handles shapes from 1D to 8D") {
    val data = Array(
      3.14, 2.718, 6.626,
      1.618, 0.577, 4.669,
      2.502, 1.414, 1.732
    ) // 9 elements

    val shapes = List(
      List(9),                  // 1D
      List(3, 3),               // 2D
      List(1, 3, 3),            // 3D
      List(1, 1, 3, 3),         // 4D
      List(1, 1, 1, 3, 3),      // 5D
      List(1, 1, 1, 1, 3, 3),   // 6D
      List(1, 1, 1, 1, 1, 3, 3),// 7D
      List(1, 1, 1, 1, 1, 1, 3, 3) // 8D
    )

    for (shape <- shapes) {
      val t = Tensor(data, shape)

      // Confirm shape
      assert(t.shape == shape)

      // Confirm first and last elements are correct
      val firstIdx = List.fill(shape.size)(0)
      val lastIdx = shape.indices.map(i => shape(i) - 1).toList

      assert(t(firstIdx: _*) == data(0), s"Failed at shape $shape: first element mismatch")
      assert(t(lastIdx: _*) == data.last, s"Failed at shape $shape: last element mismatch")

      // Reshape to 1D and verify all elements match
      val flat = t.reshape(data.length)
      for (i <- data.indices) {
        assert(flat(i) == data(i), s"Reshape failed at index $i for shape $shape")
      }
    }
  }

  test("Create Tensor from HDF Dataset") {
    val knownShape = List(16, 16, 3)
    val t = Tensor[Float](d)
    val tInt = t.map(_.round.toInt)
    // Reshape to 1D
    val tIntReshape = tInt.reshape(knownShape.product)
    val tIntList = tIntReshape.toList.asInstanceOf[List[Int]]

    assert(t.shape == knownShape)
    // We have enumerated each value to index so this should be a good check.
    // +1/-1 to account for rounding errors.
    val eqCond = (f: Float, i: Int) => f == i || f == i - 1 || f == i + 1
    for (i <- 0 until knownShape.product) assert(eqCond(tIntReshape(i), i), s"At Tensor index $i expected $i +-1 but got ${tIntReshape(i)}")
    for (i <- 0 until knownShape.product) assert(eqCond(tIntList(i), i), s"At List index $i expected $i +-1 but got ${tIntReshape(i)}")
  }

  test("Access all elements of Tensor") {
    val td = Tensor(Array(1, 2, 3, 4, 5, 6, 7, 8), List(2, 2, 2))
    val shape = td.shape

    var counter = 1
    for {
      i <- 0 until shape(0)
      j <- 0 until shape(1)
      k <- 0 until shape(2)
    } {
      assert(td(i, j, k) == counter)
      counter += 1
    }
  }

  test("Import from JSON") {
    val json =
      """
        [
          [
            [
              [1.0, 2.0],
              [3.0, 4.0]
            ],
            [
              [5.0, 6.0],
              [7.0, 8.0]
            ]
          ]
        ]
      """

    val decoded = decode[Tensor[Double]](json)

    assert(decoded.isRight, "JSON should decode successfully")

    val expectedFlat = List(List(List(List(1.0, 2.0), List(3.0, 4.0)), List(List(5.0, 6.0), List(7.0, 8.0))))

    decoded match {
      case Right(tensor: Tensor[Double]) =>
        assert(tensor.toList == expectedFlat, "Flat data should match")
      case Left(err) =>
        fail(s"Decoding failed: $err")
    }
  }

  test("Saving the file") {
    val td = Tensor(
      Array(3.14, 2.718, 6.626, 1.618, 2.122,
            0.577, 4.669, 2.502, 1.414, 1.732),
      List(2, 5))

    td.save("./test.txt")
  }

  test("From Numpy") {
    val td = Tensor.fromNumpy[Double]("src/test/scala/nir/samples/fashnion-mnist-sample.npy")

    println(td.shape)
  }

  test("toList produces correct nested structure") {
    val data = Array(1, 2, 3, 4, 5, 6)
    val t = Tensor(data, List(2, 3))
    val expected = List(List(1, 2, 3), List(4, 5, 6))
    assertEquals(t.toList, expected)
  }

  test("squeeze removes singleton dimensions from left") {
    val data = Array(1, 2, 3, 4)
    val t = Tensor(data, List(1, 1, 2, 2))
    val squeezed = t.squeeze
    assertEquals(squeezed.shape, List(2, 2))
    assertEquals(squeezed(0, 0), 1)
    assertEquals(squeezed(1, 1), 4)
  }
}
