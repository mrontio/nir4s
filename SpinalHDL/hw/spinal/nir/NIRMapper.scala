package nir

import java.io.File
import io.jhdf.HdfFile
import io.jhdf.api.{Attribute, Dataset, Group, Node}
import scala.jdk.CollectionConverters._

object NIRMapper {
  def loadGraph(f: File, nodePath: String = "/node/nodes", edgePath: String = "/node/edges") {
        if (!f.exists()) {
      Console.err.println(s"ERROR: file not found: ${f.getAbsolutePath}")
      sys.exit(2)
    }

    val hdf = new HdfFile(f)
    val rawNodes = loadNodes(hdf)


  private def loadRawNodes(hdf: HdfFile): Set[RawNode] = {
    def loadEdges(edgeHDF: Node): Set[(String, String)] = edgeHDF match {
      case dst: Dataset =>
        dst.getData().asInstanceOf[Array[Array[String]]].map { case Array(a, b) => (a, b) }.toSet
    }

    val edges = loadEdges(hdf.getByPath(edgePath))
    val nodeHDF = hdf.getByPath(nodePath)

    val rawNode = nodeHDF match {
      case grp: Group =>
        // grp.getChildren: java.util.Map[String, Node]
        grp.getChildren.asScala.collect {
          case (name, childGrp: Group) => {
            parseNode(childGrp, edges)
          }
        }.toSet
      case other =>
        throw new IllegalArgumentException(
          s"Expected a Group at '$nodePath', but found ${other.getClass.getName}"
        )
    }
  }



  private def parseNode(node: Group, edges: Set[(String, String)]): NIRNode = {
    // 1) Collect all attributes into a Map[name -> rawData]
    val attrs: Map[String, Any] =
      node.getChildren.asScala
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
    val params = attrs("type").asInstanceOf[String] match {
      case "Input" =>
        InputParams(
          shape = get1DLong("shape"),
        )

      case "Output" =>
        OutputParams(
          shape      = get1DLong("shape"),
        )
      case "LIF" =>
        LIFParams(
          tau         = get1DFloat("tau"),
          r           = get1DFloat("r"),
          v_leak      = get1DFloat("v_leak"),
          v_threshold = get1DFloat("v_threshold"),
        )

      case "CubaLIF" =>
        CubaLIFParams(
          tau        = get1DFloat("tau"),
          tauSynExc  = get1DFloat("tauSynExc"),
          tauSynInh  = get1DFloat("tauSynInh"),
        )

      case "Linear" =>
        LinearParams(
          weight      = get1DFloat("weight"),
        )

      case "LI" =>
        LIParams(
          tau = get1DFloat("tau"),
          r = get1DFloat("r"),
          v_leak = get1DFloat("v_leak"),
        )

      case "Affine" =>
        AffineParams(
          bias        = get1DFloat("bias"),
          weight      = get2DFloat("weight"),
        )

      case other =>
        throw new UnsupportedOperationException(
          s"NIRMapper: Unknown nodeType '$other' in group '${node.getName}'"
        )
    }

    RawNode(
      id       = node.getName,
      previous = getIncomingEdges(node.getName),
      params   = params
    )
  }
}
