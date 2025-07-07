package nir

import java.io.File
import io.jhdf.HdfFile
import io.jhdf.api.{Attribute, Dataset, Group, Node}
import scala.jdk.CollectionConverters._

object NIRMapper {
  def loadNodes(f: File, nodePath: String = "/node/nodes", edgePath: String = "/node/edges"): Set[NIRNode] = {
    def loadEdges(edgeHDF: Node): Set[(String, String)] = edgeHDF match {
      case dst: Dataset =>
        dst.getData().asInstanceOf[Array[Array[String]]].map { case Array(a, b) => (a, b) }.toSet
    }

    if (!f.exists()) {
      Console.err.println(s"ERROR: file not found: ${f.getAbsolutePath}")
      sys.exit(2)
    }

    val hdf = new HdfFile(f)
    val edges = loadEdges(hdf.getByPath(edgePath))
    val nodeHDF = hdf.getByPath(nodePath)

    nodeHDF match {
      case grp: Group =>
        // grp.getChildren: java.util.Map[String, Node]
        grp.getChildren.asScala.collect {
          case (name, childGrp: Group) => {
            parseNodeGroup(childGrp, edges)
          }
        }.toSet
      case other =>
        throw new IllegalArgumentException(
          s"Expected a Group at '$nodePath', but found ${other.getClass.getName}"
        )
    }
  }



  private def parseNodeGroup(grp: Group, edges: Set[(String, String)]): NIRNode = {
    // 1) Collect all attributes into a Map[name -> rawData]
    val attrs: Map[String, Any] =
      grp.getChildren.asScala
        .map { case (attrName, attr: Dataset) => attrName -> attr.getData }
        .toMap


    // 2) Helpers to extract typed values
    def getStr(key: String): String    = attrs(key).asInstanceOf[String]
    def getInt(key: String): Int       = attrs(key).asInstanceOf[Int]
    def getFloat(key: String): Float = attrs(key).asInstanceOf[Float]


    def get1DLong(key: String): Array[Long] = attrs(key).asInstanceOf[Array[Long]]
    def get1DFloat(key: String): Array[Float] = attrs(key) match {
      case f: Float => Array(f)
      case arr: Array[Float] => arr
    }
    def get2DFloat(key: String): Array[Array[Float]] = attrs(key).asInstanceOf[Array[Array[Float]]]

    def getIncomingEdges(node: String): Set[String] =
      edges.collect { case (p, n) if n == node => p }


    // 4) Dispatch based on the "nodeType" attribute
    attrs("type").asInstanceOf[String] match {
      case "Input" =>
        Input(
          id          = grp.getName,
          shape = get1DLong("shape"),
          previous = Set.empty[String]

        )

      case "Output" =>
        Output(
          id         = grp.getName,
          shape      = get1DLong("shape"),
          previous   = getIncomingEdges(grp.getName)
        )
      case "LIF" =>
        LIF(
          id          = grp.getName,
          tau         = get1DFloat("tau"),
          r           = get1DFloat("r"),
          v_leak      = get1DFloat("v_leak"),
          v_threshold = get1DFloat("v_threshold"),
          previous   = getIncomingEdges(grp.getName)
        )

      case "CubaLIF" =>
        CubaLIF(
          id         = grp.getName,
          tau        = get1DFloat("tau"),
          tauSynExc  = get1DFloat("tauSynExc"),
          tauSynInh  = get1DFloat("tauSynInh"),
          previous   = getIncomingEdges(grp.getName)
        )

      case "Linear" =>
        Linear(
          id          = grp.getName,
          weight      = get1DFloat("weight"),
          previous   = getIncomingEdges(grp.getName)
        )

      case "LI" =>
        LI(
          id = grp.getName,
          tau = get1DFloat("tau"),
          r = get1DFloat("r"),
          v_leak = get1DFloat("v_leak"),
          previous = getIncomingEdges(grp.getName)
        )

      case "Affine" =>
        Affine(
          id          = grp.getName,
          bias        = get1DFloat("bias"),
          weight      = get2DFloat("weight"),
          previous   = getIncomingEdges(grp.getName)
        )

      case other =>
        throw new UnsupportedOperationException(
          s"NIRMapper: Unknown nodeType '$other' in group '${grp.getName}'"
        )
    }
  }
}
