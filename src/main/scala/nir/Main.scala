package nir

import java.io.File
import io.jhdf.HdfFile
import io.jhdf.api.{Attribute, Dataset, Group, Node}
import scala.jdk.CollectionConverters._

object Main extends App {
  val default = "src/test/scala/nir/samples/conv1/network.nir"
  var path = default
  if (args.length != 1) {
    println(s"Using default $default")
    path = default
  }
  val file = new File(path)
  val hdf = new HdfFile(file)
  val nodeHDF = hdf.getByPath("/node/nodes")
  val d: Dataset = nodeHDF match {
    case g1: Group => g1.getChild("0") match {
      case g2: Group => g2.getChild("weight").asInstanceOf[Dataset]
    }
  }

  // val f = Fensor(d)
  val f = Fensor(Array(1, 2, 3, 4, 5, 6), List(2, 3))

  println(f.toList)

}
