package nir

import munit.FunSuite
import java.io.File
import java.nio.file.{Files, Paths}
import io.jhdf.HdfFile
import io.jhdf.api.{Attribute, Dataset, Group, Node}
import scala.jdk.CollectionConverters._

import tensor.{TensorDynamic}

class TensorStaticSpec extends FunSuite {
  test("Create TensorStatic objects from TensorDynamic list") {
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
    }
  }
}
