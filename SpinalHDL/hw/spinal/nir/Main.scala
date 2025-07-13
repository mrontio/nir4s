package nir

import java.io.File

object Main extends App {
  if (args.length != 1) {
    Console.err.println("Usage: sbt \"run <path/to/file.h5>\"")
    sys.exit(1)
  }
  val file = new File(args(0))

  val g: NIRGraph = NIRMapper.loadGraph(file)
  println(s"top: ${g.input}")
  println(s"top previous: " + g.input.previous.mkString(", "))
  println(s"top: ${g.output}")
  println(s"bot previous: " + g.output.previous.mkString(", "))

}
