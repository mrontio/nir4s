package nir

import java.io.File

object Main extends App {
  if (args.length != 1) {
    Console.err.println("Usage: sbt \"run <path/to/file.h5>\"")
    sys.exit(1)
  }
  val file = new File(args(0))

  val g = NIRGraph(NIRMapper.loadNodes(file))
  print(g.bot.previous.mkString(", "))
  print(g.bot)
}
