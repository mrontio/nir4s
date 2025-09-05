package nir

import java.io.File
import io.jhdf.HdfFile
import io.jhdf.api.{Attribute, Dataset, Group, Node}

object Main extends App {
  val default = "src/test/scala/nir/samples/conv1/network.nir"
  var path = default
  if (args.length != 1) {
    println(s"Using default $default")
    path = default
  }



}
