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

      case "IF" =>
        IFParams(
          r           = Tensor.fromHDFMap("r", attrs),
          v_reset      = Tensor.fromHDFMap("v_reset", attrs),
          v_threshold = Tensor.fromHDFMap("v_threshold", attrs),
        )

      case "LIF" =>
        LIFParams(
          tau         = Tensor.fromHDFMap("tau", attrs),
          r           = Tensor.fromHDFMap("r", attrs),
          v_leak      = Tensor.fromHDFMap("v_leak", attrs),
          v_threshold = Tensor.fromHDFMap("v_threshold", attrs),
        )


      case "CubaLIF" =>
        CubaLIFParams(
          tau        = Tensor.fromHDFMap("tau", attrs),
          tauSynExc  = Tensor.fromHDFMap("tauSynExc", attrs),
          tauSynInh  = Tensor.fromHDFMap("tauSynInh", attrs),
        )

      case "Linear" =>
        LinearParams(
          weight      = Tensor.fromHDFMap("weight", attrs),
        )

      case "LI" =>
        LIParams(
          tau = Tensor.fromHDFMap("tau", attrs),
          r = Tensor.fromHDFMap("r", attrs),
          v_leak = Tensor.fromHDFMap("v_leak", attrs),
        )

      case "Affine" =>
        AffineParams(
          bias        = Tensor.fromHDFMap("bias", attrs),
          weight      = Tensor.fromHDFMap("weight", attrs),
        )

      case "Conv1d" =>
        Conv1DParams(
          weights = Conv1DWeights(
            get = Tensor.fromHDFMap[Float]("weight", attrs).asInstanceOf[Tensor3D[Float]],
          ),
          bias = Tensor.fromHDFMap[Float]("bias", attrs).asInstanceOf[Tensor1D[Float]],
          stride = getAttr[Array[Long]]("stride"),
          padding = getAttr[Array[Long]]("padding"),
          dilation = getAttr[Array[Long]]("dilation"),
          groups = getAttr[Long]("groups"),
          input_shape = getAttr[Long]("input_shape")
        )

      case "Flatten" =>
        FlattenParams(
          start_dim = getAttr[Long]("start_dim"),
          end_dim = getAttr[Long]("end_dim"),
          input_type = Tensor.fromHDFMap[Long]("input_type", attrs)
        )


      case other =>
        throw new UnsupportedOperationException(
          s"NIRMapper: nodeType '$other' in node '${node.getName}' not yet supported.'"
        )
    }

    RawNode(
      id       = node.getName,
      prevIds = getIncomingEdges(node.getName),
      params   = params
    )
  }
}
