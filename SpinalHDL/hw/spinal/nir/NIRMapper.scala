package nir

import java.io.File
import io.jhdf.HdfFile
import io.jhdf.api.{Attribute, Dataset, Group, Node}
import scala.jdk.CollectionConverters._

object NIRMapper {
  def loadGraph(f: File, nodePath: String = "/node/nodes", edgePath: String = "/node/edges"): NIRGraph = {
    if (!f.exists()) {
      Console.err.println(s"ERROR: file not found: ${f.getAbsolutePath}")
      sys.exit(2)
    }

    val hdf = new HdfFile(f)
    val nodeHDF = hdf.getByPath(nodePath)
    val edges = loadEdges(hdf.getByPath(edgePath))
    NIRGraph.fromRaw(loadRawNodes(nodeHDF, edges))

  }

  private def loadEdges(edgeHDF: Node): Set[(String, String)] = edgeHDF match {
    case dst: Dataset =>
      dst.getData().asInstanceOf[Array[Array[String]]].map { case Array(a, b) => (a, b) }.toSet
  }

  private def loadRawNodes(nodeHDF: Node, edges: Set[(String, String)]): Set[RawNode] = {
    nodeHDF match {
      case grp: Group =>
        grp.getChildren.asScala.collect {
          case (_, childGrp: Group) =>
            parseNode(childGrp, edges)
        }.toSet.asInstanceOf[Set[RawNode]]
      case other =>
        throw new IllegalArgumentException(
          s"Expected a Group, but found ${other.getClass.getName}"
        )
    }
  }



  private def parseNode(node: Group, edges: Set[(String, String)]): RawNode = {
    // 1) Collect all attributes into a Map[name -> rawData]
    val attrs: Map[String, Any] =
      node.getChildren.asScala
        .map { case (attrName, attr: Dataset) => attrName -> attr.getData }
        .toMap


    // 2) Helpers to extract typed values
    // Scalars
    def getStr(key: String): String    = attrs(key).asInstanceOf[String]
    def getInt(key: String): Int       = attrs(key).asInstanceOf[Int]
    def getFloat(key: String): Float   = attrs(key).asInstanceOf[Float]

    // Vectors
    def get1DLong(key: String): Array[Long]   = attrs(key).asInstanceOf[Array[Long]]
    def get1DFloat(key: String): Array[Float] = attrs(key) match {
      case f: Float => Array(f)
      case arr: Array[Float] => arr
    }

    def getAttr[T](key: String): T = attrs(key).asInstanceOf[T]

    def getIncomingEdges(node: String): Set[String] =
      edges.collect { case (p, n) if n == node => p }


    // 4) Dispatch based on the "nodeType" attribute
    val params = attrs("type").asInstanceOf[String] match {
      case "Input" =>
        InputParams(
          shape = get1DLong("shape").map(_.toInt).toArray,
        )

      case "Output" =>
        OutputParams(
          shape      = get1DLong("shape").map(_.toInt).toArray,
        )
      case "LIF" =>
        LIFParams(
          tau         = getAttr[nir.Matrix1D[Float]]("tau"),
          r           = getAttr[nir.Matrix1D[Float]]("r"),
          v_leak      = getAttr[nir.Matrix1D[Float]]("v_leak"),
          v_threshold = getAttr[nir.Matrix1D[Float]]("v_threshold"),
        )


      case "CubaLIF" =>
        CubaLIFParams(
          tau        = getAttr[nir.Matrix1D[Float]]("tau"),
          tauSynExc  = getAttr[nir.Matrix1D[Float]]("tauSynExc"),
          tauSynInh  = getAttr[nir.Matrix1D[Float]]("tauSynInh"),
        )

      case "Linear" =>
        LinearParams(
          weight      = getAttr[nir.Matrix1D[Float]]("weight"),
        )

      case "LI" =>
        LIParams(
          tau = getAttr[nir.Matrix1D[Float]]("tau"),
          r = getAttr[nir.Matrix1D[Float]]("r"),
          v_leak = getAttr[nir.Matrix1D[Float]]("v_leak"),
        )

      case "Affine" =>
        AffineParams(
          bias        = getAttr[nir.Matrix1D[Float]]("bias"),
          weight      = getAttr[nir.Matrix2D[Float]]("weight"),
        )

      case other =>
        throw new UnsupportedOperationException(
          s"NIRMapper: Unknown nodeType '$other' in group '${node.getName}'"
        )
    }

    RawNode(
      id       = node.getName,
      prevIds = getIncomingEdges(node.getName),
      params   = params
    )
  }
}
