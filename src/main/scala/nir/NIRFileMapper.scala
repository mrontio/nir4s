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
      throw new java.io.FileNotFoundException(s"ERROR: file not found: ${f.getAbsolutePath}")
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
          shape = Tensor[Long](getDataset("shape"))
        )

      case "Output" =>
        OutputParams(
          shape = Tensor[Long](getDataset("shape"))
        )

      case "I" =>
        IParams(
          r = Tensor[Double](getDataset("r")).map(_.toFloat)
        )

      case "IF" =>
        IFParams(
          r           = Tensor[Float](getDataset("r")),
          v_reset     = Tensor[Float](getDataset("v_reset")),
          v_threshold = Tensor[Float](getDataset("v_threshold"))
        )

      case "LIF" =>
        LIFParams(
          tau         = Tensor[Float](getDataset("tau")),
          r           = Tensor[Float](getDataset("r")),
          v_leak      = Tensor[Float](getDataset("v_leak")),
          v_threshold = Tensor[Float](getDataset("v_threshold"))
        )

      case "CubaLIF" =>
        CubaLIFParams(
          tau       = Tensor[Float](getDataset("tau")),
          tauSynExc = Tensor[Float](getDataset("tauSynExc")),
          tauSynInh = Tensor[Float](getDataset("tauSynInh"))
        )

      case "Linear" =>
        LinearParams(
          weight = Tensor[Float](getDataset("weight"))
        )

      case "LI" =>
        LIParams(
          tau    = Tensor[Float](getDataset("tau")),
          r      = Tensor[Float](getDataset("r")),
          v_leak = Tensor[Float](getDataset("v_leak"))
        )

      case "Affine" =>
        AffineParams(
          bias   = Tensor[Float](getDataset("bias")),
          weight = Tensor[Float](getDataset("weight"))
        )

      case "Conv1d" =>
        Conv1DParams(
          weight      = Tensor[Float](getDataset("weight")),
          bias        = Tensor[Float](getDataset("bias")),
          stride      = Tensor[Long](getDataset("stride")),
          padding     = Tensor[Long](getDataset("padding")),
          dilation    = Tensor[Long](getDataset("dilation")),
          groups      = getData[Long]("groups"),
          input_shape = getData[Long]("input_shape")
        )

      case "Conv2d" =>
        Conv2DParams(
          weight      = Tensor[Float](getDataset("weight")),
          bias        = Tensor[Float](getDataset("bias")),
          stride      = Tensor[Long](getDataset("stride")),
          padding     = Tensor[Long](getDataset("padding")),
          dilation    = Tensor[Long](getDataset("dilation")),
          groups      = getData[Long]("groups"),
          input_shape = Tensor[Long](getDataset("input_shape"))
        )

      case "Flatten" =>
        FlattenParams(
          start_dim  = getData[Long]("start_dim"),
          end_dim    = getData[Long]("end_dim"),
          input_type = Tensor[Long](getDataset("input_type"))
        )

      case "SumPool2d" =>
        SumPool2DParams(
          kernel_size = Tensor[Long](getDataset("kernel_size")),
          padding     = Tensor[Long](getDataset("padding")),
          stride      = Tensor[Long](getDataset("stride"))
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
