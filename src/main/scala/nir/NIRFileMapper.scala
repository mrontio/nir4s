package nir

import java.io.File
import io.jhdf.HdfFile
import scala.reflect.ClassTag
import io.jhdf.api.{Attribute, Dataset, Group, Node}
import scala.jdk.CollectionConverters._

object NIRFileMapper {
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
            parseNode(childGrp, edges).asInstanceOf[RawNode]
        }.toSet
      case other =>
        throw new IllegalArgumentException(
          s"Expected a Group, but found ${other.getClass.getName}"
        )
    }
  }

  private def parseNode(node: Group, edges: Set[(String, String)]): RawNode = {
    def getIncomingEdges(nodeName: String): Set[String] =
      edges.collect { case (p, n) if n == nodeName => p }

    def getDataset(attr: String): Dataset = node.getChild(attr) match {
      case d: Dataset => d
      case g: Group   => throw new java.text.ParseException(s"In Group ${node.getName()}, expected $attr to be of type Dataset but it is of type Group.", 0)
    }

    def getData[T: ClassTag](attr: String): T =
      getDataset(attr).getData match {
        case x: T => x
        case o    => throw new java.text.ParseException(s"In Group  ${node.getName()} $attr is not of expected type.", 0)
      }


    val params = getData[String]("type") match {
      case "Input" =>
        InputParams(
          shape      = Tensor(getDataset("shape")).asInstanceOf[Tensor[Long]]
        )

      case "Output" =>
        OutputParams(
          shape      = Tensor(getDataset("shape")).asInstanceOf[Tensor[Long]]
        )

      case "IF" =>
        IFParams(
          r           = Tensor(getDataset("r")).asInstanceOf[Tensor[Float]],
          v_reset      = Tensor(getDataset("v_reset")).asInstanceOf[Tensor[Float]],
          v_threshold = Tensor(getDataset("v_threshold")).asInstanceOf[Tensor[Float]]
        )

      case "LIF" =>
        LIFParams(
          tau         = Tensor(getDataset("tau")).asInstanceOf[Tensor[Float]],
          r           = Tensor(getDataset("r")).asInstanceOf[Tensor[Float]],
          v_leak      = Tensor(getDataset("v_leak")).asInstanceOf[Tensor[Float]],
          v_threshold = Tensor(getDataset("v_threshold")).asInstanceOf[Tensor[Float]]
        )

      case "CubaLIF" =>
        CubaLIFParams(
          tau        = Tensor(getDataset("tau")).asInstanceOf[Tensor[Float]],
          tauSynExc  = Tensor(getDataset("tauSynExc")).asInstanceOf[Tensor[Float]],
          tauSynInh  = Tensor(getDataset("tauSynInh")).asInstanceOf[Tensor[Float]]
        )

      case "Linear" =>
        LinearParams(
          weight = Tensor(getDataset("weight")).asInstanceOf[Tensor[Float]]
        )

      case "LI" =>
        LIParams(
          tau = Tensor(getDataset("tau")).asInstanceOf[Tensor[Float]],
          r = Tensor(getDataset("r")).asInstanceOf[Tensor[Float]],
          v_leak = Tensor(getDataset("v_leak")).asInstanceOf[Tensor[Float]]
        )

      case "Affine" =>
        AffineParams(
          bias        = Tensor(getDataset("bias")).asInstanceOf[Tensor[Float]],
          weight      = Tensor(getDataset("weight")).asInstanceOf[Tensor[Float]]
        )

      case "Conv1d" =>
        Conv1DParams(
          weight = Tensor(getDataset("weight")).asInstanceOf[Tensor[Float]],
          bias = Tensor(getDataset("bias")).asInstanceOf[Tensor[Float]],
          stride = Tensor(getDataset("stride")).asInstanceOf[Tensor[Long]],
          padding = Tensor(getDataset("padding")).asInstanceOf[Tensor[Long]],
          dilation = Tensor(getDataset("dilation")).asInstanceOf[Tensor[Long]],
          groups = getData[Long]("groups"),
          input_shape = getData[Long]("input_shape")
        )

      case "Conv2d" =>
        Conv2DParams(
          weight = Tensor(getDataset("weight")).asInstanceOf[Tensor[Float]],
          bias = Tensor(getDataset("bias")).asInstanceOf[Tensor[Float]],
          stride = Tensor(getDataset("stride")).asInstanceOf[Tensor[Long]],
          padding = Tensor(getDataset("padding")).asInstanceOf[Tensor[Long]],
          dilation = Tensor(getDataset("dilation")).asInstanceOf[Tensor[Long]],
          groups = getData[Long]("groups"),
          input_shape = Tensor(getDataset("input_shape")).asInstanceOf[Tensor[Long]]
        )

      case "Flatten" =>
        FlattenParams(
          start_dim = getData[Long]("start_dim"),
          end_dim = getData[Long]("end_dim"),
          input_type = Tensor(getDataset("input_type")).asInstanceOf[Tensor[Long]]
        )

      case "SumPool2d" =>
        SumPool2DParams(
          kernel_size = Tensor(getDataset("kernel_size")).asInstanceOf[Tensor[Long]],
          padding = Tensor(getDataset("padding")).asInstanceOf[Tensor[Long]],
          stride = Tensor(getDataset("stride")).asInstanceOf[Tensor[Long]]
        )

      case other =>
        throw new UnsupportedOperationException(
          s"NIRFileMapper: nodeType '$other' in node '${node.getName}' not yet supported.'"
        )
    }

    RawNode(
      id       = node.getName,
      prevIds  = getIncomingEdges(node.getName),
      params   = params.asInstanceOf[NIRParams]
    )
  }
}
