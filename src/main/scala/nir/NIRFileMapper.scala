package nir

import java.io.File
import io.jhdf.HdfFile
import scala.reflect.ClassTag
import io.jhdf.api.{Attribute, Dataset, Group, Node}
import scala.jdk.CollectionConverters._

import tensor._

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
          shape      = TensorDynamic[Long](getDataset("shape")).squeeze.toStatic
        )

      case "Output" =>
        OutputParams(
          shape      = TensorDynamic[Long](getDataset("shape")).squeeze.toStatic
        )


      case "IF" =>
        IFParams(
          r           = TensorDynamic[Float](getDataset("r")).squeeze.toStatic,
          v_reset      = TensorDynamic[Float](getDataset("v_reset")).squeeze.toStatic,
          v_threshold = TensorDynamic[Float](getDataset("v_threshold")).squeeze.toStatic
        )

      case "LIF" =>
        LIFParams(
          tau         = TensorDynamic[Float](getDataset("tau")).squeeze.toStatic,
          r           = TensorDynamic[Float](getDataset("r")).squeeze.toStatic,
          v_leak      = TensorDynamic[Float](getDataset("v_leak")).squeeze.toStatic,
          v_threshold = TensorDynamic[Float](getDataset("v_threshold")).squeeze.toStatic
        )

      case "CubaLIF" =>
        CubaLIFParams(
          tau        = TensorDynamic[Float](getDataset("tau")).squeeze.toStatic,
          tauSynExc  = TensorDynamic[Float](getDataset("tauSynExc")).squeeze.toStatic,
          tauSynInh  = TensorDynamic[Float](getDataset("tauSynInh")).squeeze.toStatic
        )

      case "Linear" =>
        LinearParams(
          weight = TensorDynamic[Float](getDataset("weight")).squeeze.toStatic
        )

      case "LI" =>
        LIParams(
          tau = TensorDynamic[Float](getDataset("tau")).squeeze.toStatic,
          r = TensorDynamic[Float](getDataset("r")).squeeze.toStatic,
          v_leak = TensorDynamic[Float](getDataset("v_leak")).squeeze.toStatic
        )

      case "Affine" =>
        AffineParams(
          bias        = TensorDynamic[Float](getDataset("bias")).squeeze.toStatic,
          weight      = TensorDynamic[Float](getDataset("weight")).squeeze.toStatic
        )

      case "Conv1d" =>
        Conv1DParams(
          weight = TensorDynamic[Float](getDataset("weight")).squeeze.toStatic,
          bias = TensorDynamic[Float](getDataset("bias")).squeeze.toStatic,
          stride = TensorDynamic[Long](getDataset("stride")).squeeze.toStatic,
          padding = TensorDynamic[Long](getDataset("padding")).squeeze.toStatic,
          dilation = TensorDynamic[Long](getDataset("dilation")).squeeze.toStatic,
          groups = getData[Long]("groups"),
          input_shape = getData[Long]("input_shape")
        )

      case "Conv2d" =>
        Conv2DParams(
          weight = TensorDynamic[Float](getDataset("weight")).squeeze.toStatic,
          bias = TensorDynamic[Float](getDataset("bias")).squeeze.toStatic,
          stride = TensorDynamic[Long](getDataset("stride")).squeeze.toStatic,
          padding = TensorDynamic[Long](getDataset("padding")).squeeze.toStatic,
          dilation = TensorDynamic[Long](getDataset("dilation")).squeeze.toStatic,
          groups = getData[Long]("groups"),
          input_shape = TensorDynamic[Long](getDataset("input_shape")).squeeze.toStatic
        )

      case "Flatten" =>
        FlattenParams(
          start_dim = getData[Long]("start_dim"),
          end_dim = getData[Long]("end_dim"),
          input_type = TensorDynamic[Long](getDataset("input_type")).squeeze.toStatic
        )

      case "SumPool2d" =>
        SumPool2DParams(
          kernel_size = TensorDynamic[Long](getDataset("kernel_size")).squeeze.toStatic,
            padding = TensorDynamic[Long](getDataset("padding")).squeeze.toStatic,
            stride = TensorDynamic[Long](getDataset("stride")).squeeze.toStatic
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
