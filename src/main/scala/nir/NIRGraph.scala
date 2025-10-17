package nir

import java.io.File

/** Directed graph composed of NIR nodes.
  *
  * @param nodes collection of nodes forming the graph
  */
case class NIRGraph(nodes: Set[NIRNode]) {
  val input: NIRNode = nodes.find(_.previous.size == 0).get
  val output: NIRNode = nodes.find(_.params.nirType == "Output")
    .getOrElse(throw new Exception("NIR Graph does not contain node type 'Output', please re-generate."))
  val nodeMap: Map[String, NIRNode] = nodes.map(node => node.id -> node).toMap


  def getNode(id: String): NIRNode = nodeMap(id)
  override def toString: String = NIRGraph.getStructureString(output)
}

object NIRGraph {
  def fromSimpleChain(input: InputParams, chain: List[NIRParams], output: OutputParams): NIRGraph = {
    val inputNode = RawNode("input", Set(), input)

    val withId = chain.zipWithIndex.map{ case (n, idx) => (n, idx.toString)} :+ (output, "output")

    val prevId = "input" :: withId.map(_._2)

    val rawNodes = inputNode :: withId.zip(prevId).map{ case ((params, id), prevId) => RawNode(id, Set(prevId), params)}

    fromRaw(rawNodes.toSet)
  }

  def reduceConv2DIFSubgraph(graph: NIRGraph): NIRGraph = {
    // Step 1: Find Conv2D nodes that have IF as previous
    val ifAfterConv = graph.nodes.filter { node =>
      node.params.nirType == "IF" &&
      node.previous.nonEmpty &&
      node.previous.head.params.nirType == "Conv2d"
    }

    if (ifAfterConv.isEmpty) {
      return graph
    }


    // Step 2: Get the matched nodes
    val ifNode = ifAfterConv.head                  // The IF node
    val convNode = ifNode.previous.head            // The Conv node (IF.previous)
    val nodeBeforeConv = convNode.previous.head    // Node before Conv (Conv.previous = Input)
    val nodeAfterIF = graph.nodes.filter(_.previous.contains(ifNode)).head

    // Step 3: Extract Conv2DParams and create Conv2DIFParams
    val conv2dParams = convNode.params.asInstanceOf[Conv2DParams]
    val fusedParams: NIRParams = Conv2DIFParams(
      weight = conv2dParams.weight,
      bias = conv2dParams.bias,
      stride = conv2dParams.stride,
      padding = conv2dParams.padding,
      dilation = conv2dParams.dilation,
      groups = conv2dParams.groups,
      input_shape = conv2dParams.input_shape
    )

    // Step 4: Create the fused subgraph node
    val fusedNode = NIRNode(
      id = s"fused_${convNode.id}_${ifNode.id}",
      previous = Set(nodeBeforeConv),
      params = fusedParams
    )

    // Step 5: Rebuild the graph - remove IF and Conv, add fused node
    val newNodes = graph.nodes.flatMap { node =>
      if (node.id == ifNode.id || node.id == convNode.id) {
        None  // Remove both nodes
      } else if (nodeAfterIF == node) {
        node.previous = Set(fusedNode)
        Some(node)
      } else {
        Some(node)
      }
    }
    // Step 6: Return new graph
    new NIRGraph(newNodes + fusedNode)
  }

  def fromRaw(rawNodes: Set[RawNode]): NIRGraph = {
    val topRaw = rawNodes.find(_.params.nirType == "Input")
      .getOrElse(throw new Exception("NIR Graph does not contain node type 'Input', please re-generate."))

    val top: NIRNode = NIRNode(topRaw.id, Set.empty[NIRNode], topRaw.params)
    // TODO: put initial case into recursive function
    val nodes = Set(top) ++ convertRawNodes(rawNodes, top)
    new NIRGraph(nodes)
  }

  def apply(f: File): NIRGraph = {
    NIRFileMapper.loadGraph(f)
  }

  private def convertRawNodes(nodes: Set[RawNode], top: NIRNode): Set[NIRNode] = {
    nodes.collectFirst {
      case n if n.prevIds.contains(top.id) =>
        n
    } match {
      case Some(n) if n.id == "output" =>
        Set(NIRNode(n.id, Set(top), n.params))
      case Some(n) if n.id == top.id =>
        Set(top)
      case Some(n) =>
        val newNode = NIRNode(n.id, Set(top), n.params)
        convertRawNodes(nodes, newNode) + newNode
      case None =>
        throw new RuntimeException("error: multiple connections not yet supported or no connections found.")
    }
  }

  private def getStructureString(node: NIRNode): String = {
    node.params match {
      case i: InputParams => s"${node.id}: $i"
      case x => node.previous.collectFirst {
        case node: NIRNode => getStructureString(node)
      } + s"${node.id}: $x"
    }
  }
}
