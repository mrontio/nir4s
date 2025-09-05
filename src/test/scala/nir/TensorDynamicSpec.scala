package nir

import munit.FunSuite
import java.io.File
import java.nio.file.{Files, Paths}
import io.jhdf.HdfFile
import io.jhdf.api.{Attribute, Dataset, Group, Node}
import scala.jdk.CollectionConverters._

import tensor.{TensorDynamic}

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

  test("TensorDynamic from Array") {
    val a =
      Array(3.14, 2.718, 6.626,
        1.618, 0.577, 4.669,
        2.502, 1.414, 1.732)

    val shape = List(1, 3, 3)
    val t = TensorDynamic(a, shape)
    val tr = t.reshape(List(9))

    assert(t.shape == List(1, 3, 3))
    assert(t(0, 0, 0) == 3.14)
    assert(t(0, 2, 2) == 1.732)

    // Reshape test
    for (i <- 0 to 9) assert(tr(0) == a(0))

  }

  test("Create TensorDynamic from HDF Dataset") {
    val knownShape = List(16, 16, 3)
    val t = TensorDynamic(d).asInstanceOf[TensorDynamic[Float]]
    val tInt = t.map(_.round.toInt)
    // Reshape to 1D
    val tIntReshape = tInt.reshape(List(knownShape.product))
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
}
