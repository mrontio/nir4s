package nir

import munit.FunSuite
import java.io.File
import java.nio.file.{Files, Paths}
import io.jhdf.HdfFile
import io.jhdf.api.{Attribute, Dataset, Group, Node}
import scala.jdk.CollectionConverters._

import io.circe.parser._

import tensor._

class TensorDynamicSpec extends FunSuite {
  val conv1Path = "src/test/scala/nir/samples/conv1/network.nir"
  val file = new File(conv1Path)
  val hdf = new HdfFile(file)
  val nodeHDF = hdf.getByPath("/node/nodes")
  val d: Dataset = nodeHDF match {
    case g1: Group => g1.getChild("0") match {
      case g2: Group => g2.getChild("weight").asInstanceOf[Dataset]
    }
  }

  test("TensorDynamic handles shapes from 1D to 8D") {
    val data = Array(
      3.14, 2.718, 6.626,
      1.618, 0.577, 4.669,
      2.502, 1.414, 1.732
    ) // 9 elements

    val baseValue = data(0)
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
      val t = TensorDynamic(data, shape)

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

  test("Create TensorDynamic from HDF Dataset") {
    val knownShape = List(16, 16, 3)
    val t = TensorDynamic[Float](d)
    val tInt = t.map(_.round.toInt)
    // Reshape to 1D
    val tIntReshape = tInt.reshape(knownShape.product)
    val tIntList = tIntReshape.toList.asInstanceOf[List[Int]]

    assert(t.shape == knownShape)
    // We have enumerated each value to index so this should be a good check.
    // +1/-1 to account for rounding errors.
    val eqCond = (f: Float, i: Int) => f == i || f == i - 1 || f == i + 1
    for (i <- 0 until knownShape.product) assert(eqCond(tIntReshape(i), i), s"At TensorDynamic index $i expected $i +-1 but got ${tIntReshape(i)}")
    for (i <- 0 until knownShape.product) assert(eqCond(tIntList(i), i), s"At List index $i expected $i +-1 but got ${tIntReshape(i)}")
  }

  test("Match TensorDynamic against TensorDynamic.Rank of ranks 1..5") {
    def makeShape(rank: Int): List[Int] =
      List.fill(rank - 1)(1) :+ 6  // always ends with 6 to keep total size consistent

    val tensors = (1 to 5).map { rank =>
      val shape = makeShape(rank)
      val size = shape.product
      TensorDynamic(Array.tabulate(size)(i => i + 1), shape)
    }

    tensors.zipWithIndex.foreach { case (tensor, idx) =>
      val expectedRank = idx + 1

      val matchedRank = tensor match {
        case TensorDynamic.Rank(r) => r
        case _       => -1 // should never hit this
      }

      assert(matchedRank == expectedRank, s"Expected rank $expectedRank, got $matchedRank")
    }
  }

  test("Implicit static conversion") {
    val td = TensorDynamic(3.14, 2.718, 6.626, 1.618, 0.577, 4.669, 2.502, 1.414, 1.732)

    val ts: TensorStatic[Double] = td
  }

  test("From non-flat array") {
    val shape = List(2, 2, 2, 2)
    val arr = Array.ofDim[Int](shape(0), shape(1), shape(2), shape(3))
    TensorDynamic(TensorDynamic.flattenArray(arr), shape)
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

    val decoded = decode[TensorDynamic[Double]](json)

    assert(decoded.isRight, "JSON should decode successfully")

    val expectedFlat = List(List(List(List(1.0, 2.0), List(3.0, 4.0)), List(List(5.0, 6.0), List(7.0, 8.0))))

    decoded match {
      case Right(tensor: TensorDynamic[Double]) =>
        assert(tensor.toList == expectedFlat, "Flat data should match")
      case Left(err) =>
        fail(s"Decoding failed: $err")
    }
  }

  test("Saving the file") {
    val td = TensorDynamic(
      3.14, 2.718, 6.626, 1.618, 2.122,
      0.577, 4.669, 2.502, 1.414, 1.732)
      .reshape(2, 5)

    td.save("./test.txt")
  }
}
