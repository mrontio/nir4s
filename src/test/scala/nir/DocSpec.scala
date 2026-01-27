package nir

import munit.FunSuite
import java.io.File
import java.nio.file.{Files, Paths}
import io.jhdf.HdfFile
import io.jhdf.api.{Attribute, Dataset, Group, Node}
import scala.jdk.CollectionConverters._

import nir._
import tensor._

// Make sure that the code in the documentation runs as expected.
class DocSpec extends FunSuite {
  test("getVleak0") {
    def getVleak0(n: NIRNode): List[Float] = {
      n.params match {
        case i: InputParams => List()
        case l: LIFParams => List(l.v_leak(0))
        case _           => getVleak0(n.previous.head)
      }
    }

    val network_path = "src/test/scala/nir/samples/lif/network.nir"
    val g = NIRGraph(new File(network_path))
    val result = getVleak0(g.output)
    println(s"getVleak0 result = ${result}")

  }
}
