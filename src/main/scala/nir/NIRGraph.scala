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

def reduceAffineLIF(graph: NIRGraph): NIRGraph = {
  // Step 1: Find LIF nodes that have Affine as previous
  val lifAfterAffine = graph.nodes.filter { node =>
    node.params.nirType == "LIF" &&
    node.previous.nonEmpty &&
    node.previous.head.params.nirType == "Affine"
  }

  if (lifAfterAffine.isEmpty) {
    // No more patterns found - we're done
    return graph
  }

  // Step 2: Get the matched nodes (process first match)
  val lifNode = lifAfterAffine.head
  val affineNode = lifNode.previous.head
  val nodeBeforeAffine = affineNode.previous.head
  val nodesAfterLIF = graph.nodes.filter(_.previous.contains(lifNode))

  // Step 3: Extract parameters to create AffineIFParams
  val affineParams = affineNode.params.asInstanceOf[AffineParams]
  val lifParams = lifNode.params.asInstanceOf[LIFParams]
  val fusedParams: NIRParams = AffineLIFParams(
    weight = affineParams.weight,
    tau = lifParams.tau,
    r = lifParams.r,
    v_leak = lifParams.v_leak,
    v_threshold = lifParams.v_threshold
  )

  // Step 4: Create the fused subgraph node
  val fusedNode = NIRNode(
    id = s"fused_${affineNode.id}_${lifNode.id}",
    previous = Set(nodeBeforeAffine),
    params = fusedParams
  )

  // Step 5: Rebuild the graph - remove IF and Conv, add fused node
  val newNodes = graph.nodes.flatMap { node =>
    if (node.id == lifNode.id || node.id == affineNode.id) {
      None  // Remove both nodes
    } else if (nodesAfterLIF.contains(node)) {
      node.previous = Set(fusedNode)
      Some(node)
    } else {
      Some(node)
    }
  }

  // Step 6: Create updated graph
  val updatedGraph = new NIRGraph(newNodes + fusedNode)

  // Step 7: Recursively replace remaining patterns
  reduceAffineLIF(updatedGraph)
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
