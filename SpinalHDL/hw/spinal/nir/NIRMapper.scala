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


  private def getTensor(key: String, attrs: Map[String, Any]): Tensor = {
    val attr = attrs(key)
    attr.getClass.getName match {
      case "[F" => Tensor1D(attr.asInstanceOf[Array[Float]])
      case "[[F" => Tensor2D(attr.asInstanceOf[Array[Array[Float]]])
      case "[[[F" => Tensor3D(attr.asInstanceOf[Array[Array[Array[Float]]]])
      case "[[[[F" => Tensor4D(attr.asInstanceOf[Array[Array[Array[Array[Float]]]]])
      case a => throw new java.text.ParseException(s"Expected to read float tensor but read \"${a}\"", 0)
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
          tau         = getTensor("tau", attrs),
          r           = getTensor("r", attrs),
          v_leak      = getTensor("v_leak", attrs),
          v_threshold = getTensor("v_threshold", attrs),
        )


      case "CubaLIF" =>
        CubaLIFParams(
          tau        = getTensor("tau", attrs),
          tauSynExc  = getTensor("tauSynExc", attrs),
          tauSynInh  = getTensor("tauSynInh", attrs),
        )

      case "Linear" =>
        LinearParams(
          weight      = getTensor("weight", attrs),
        )

      case "LI" =>
        LIParams(
          tau = getTensor("tau", attrs),
          r = getTensor("r", attrs),
          v_leak = getTensor("v_leak", attrs),
        )

      case "Affine" =>
        AffineParams(
          bias        = getTensor("bias", attrs),
          weight      = getTensor("weight", attrs),
        )

      // case "Conv1d" =>
      //   Conv1DParams(
      //     weights = Conv1DWeights(
      //       get = getTensor("weight", attrs)
      //     ),
      //     bias = getTensor("bias", attrs),
      //     stride = getAttr[Array[Long]]("stride"),
      //     padding = getAttr[Array[Long]]("padding"),
      //     dilation = getAttr[Array[Long]]("dilation"),
      //     groups = getAttr[Long]("groups"),
      //     input_shape = getAttr[Long]("input_shape")
      //   )

      case other =>
        throw new UnsupportedOperationException(
          s"NIRMapper: nodeType '$other' in group '${node.getName} not yet supported.'"
        )
    }

    RawNode(
      id       = node.getName,
      prevIds = getIncomingEdges(node.getName),
      params   = params
    )
  }
}
