package nir

import munit.FunSuite
import java.io.File
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters._


class NIRSpec extends FunSuite {

  // Root directory where samples are stored
  val sampleRoot = Paths.get("src/test/scala/nir/samples")

  // Find all .nir files and group them by the first-level folder (e.g., 'li', 'foo')
  val nirFilesByGroup: Map[String, List[String]] = {
    if (Files.exists(sampleRoot)) {
      Files.walk(sampleRoot)
        .iterator()
        .asScala
        .filter(p => p.toString.endsWith("/network.nir"))
        .toList
        .groupBy { path =>
          sampleRoot.relativize(path).iterator().next().toString  // 'li', 'foo', etc.
        }
        .view
        .mapValues(_.map(_.toString))
        .toMap
    } else {
      println(s"No files found in $sampleRoot directory.")
      Map.empty
    }
  }

  // Dynamically create tests based on the folder name
  for {
    (group: String, files: List[String]) <- nirFilesByGroup
    file = new File(files(0))
  } {
    test(s"NIR file: samples/$group/network.nir") {
      print(s"yolo $group")
      val g = NIRGraph(file)

      print(g)
    }
  }


  test("Subgraph Reduction") {
    val path = sampleRoot + "/affine-lif/network.nir"
    val g = NIRGraph(new File(path))

    println(g)
    val gs = NIRGraph.reduceAffineLIF(g)
    println(gs)
  }
}
