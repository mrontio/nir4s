package nir

import munit.FunSuite
import java.io.File
import java.nio.file.{Files, Paths}
import io.jhdf.HdfFile
import io.jhdf.api.{Attribute, Dataset, Group, Node}
import scala.jdk.CollectionConverters._

import tensor.{TensorDynamic}

class TensorStaticSpec extends FunSuite {
  val conv1Path = "src/test/scala/nir/samples/conv1/network.nir"
  val file = new File(conv1Path)
  val hdf = new HdfFile(file)
  val nodeHDF = hdf.getByPath("/node/nodes")
  val d: Dataset = nodeHDF match {
    case g1: Group => g1.getChild("0") match {
      case g2: Group => g2.getChild("weight").asInstanceOf[Dataset]
    }
  }

  test("Create TensorStatic objects from TensorDynamic") {
    val dynamics = List(
      TensorDynamic(Array(1, 2, 3, 4, 5, 6), List(6)),     // 1D
      TensorDynamic(Array(1, 2, 3, 4, 5, 6), List(1, 6))   // 2D
    )

    // expected types (as predicates to avoid erasure issues on element type)
    val expect: List[Any => Boolean] = List(
      (x: Any) => x.isInstanceOf[Tensor1D[_]],
      (x: Any) => x.isInstanceOf[Tensor2D[_]]
    )

    dynamics.zip(expect).zipWithIndex.foreach { case ((td, isExpected), i) =>
      val ts = td.toStatic // should yield the correct rank-specific subtype
      assert(td.toList == ts.toList, "Failed Dynamic <-> Static equality check.")
    }


  }

  test("Read NIR weight into TensorStatic") {
    val knownShape = List(16, 16, 3)
    val td = TensorDynamic(d).asInstanceOf[TensorDynamic[Float]]
    val ts = td.toStatic

    assert(td.toList == ts.toList, "Failed Dynamic <-> Static equality check.")
  }
}
