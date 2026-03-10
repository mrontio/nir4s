package nir

import java.io.File

/** Directed graph composed of NIR nodes.
  *
  * @param nodes
  *   collection of nodes forming the graph
  */
case class NIRGraph(nodes: Set[NIRNode]) {
  val input: NIRNode = nodes.find(_.previous.size == 0).get
  val output: NIRNode = nodes
    .find(_.params.nirType == "Output")
    .getOrElse(
      throw new Exception(
        "NIR Graph does not contain node type 'Output', please re-generate."
      )
    )
  val nodeMap: Map[String, NIRNode] = nodes.map(node => node.id -> node).toMap

  def getNode(id: String): NIRNode = nodeMap(id)
  override def toString: String = NIRGraph.getStructureString(output)
}

object NIRGraph {
  def fromSimpleChain(
      input: InputParams,
      chain: List[NIRParams],
      output: OutputParams
  ): NIRGraph = {
    val inputNode = RawNode("input", Set(), input)

    val withId = chain.zipWithIndex.map { case (n, idx) =>
      (n, idx.toString)
    } :+ (output, "output")

    val prevId = "input" :: withId.map(_._2)

    val rawNodes = inputNode :: withId.zip(prevId).map {
      case ((params, id), prevId) => RawNode(id, Set(prevId), params)
    }

    fromRaw(rawNodes.toSet)
  }

  def reduce(graph: NIRGraph): NIRGraph = {
    // Step 1: Find LIF or LI nodes that have Affine as previous
    val neuronAfterAffine = graph.nodes.filter { node =>
      (node.params.nirType == "LIF" || node.params.nirType == "LI") &&
      node.previous.nonEmpty &&
      node.previous.head.params.nirType == "Affine"
    }

    if (neuronAfterAffine.isEmpty) {
      // No more patterns found - we're done
      return graph
    }

    // Step 2: Get the matched nodes (process first match)
    val neuronNode = neuronAfterAffine.head
    val affineNode = neuronNode.previous.head
    val nodeBeforeAffine = affineNode.previous.head
    val nodesAfterNeuron = graph.nodes.filter(_.previous.contains(neuronNode))

    // Step 3: Extract parameters and create fused params
    val affineParams = affineNode.params.asInstanceOf[AffineParams]
    val fusedParams: NIRParams = neuronNode.params match {
      case lif: LIFParams =>
        AffineLIFParams(
          old_linear_id = affineNode.id,
          old_lif_id = neuronNode.id,
          weight = affineParams.weight,
          tau = lif.tau,
          r = lif.r,
          v_leak = lif.v_leak,
          v_threshold = lif.v_threshold
        )
      case li: LIParams =>
        AffineLIParams(
          old_linear_id = affineNode.id,
          old_li_id = neuronNode.id,
          weight = affineParams.weight,
          tau = li.tau,
          r = li.r,
          v_leak = li.v_leak
        )
      case other =>
        throw new RuntimeException(
          s"Unexpected neuron type in reduce: ${other.nirType}"
        )
    }

    // Step 4: Create the fused subgraph node
    val fusedNode = NIRNode(
      id = s"fused_${affineNode.id}_${neuronNode.id}",
      previous = Set(nodeBeforeAffine),
      params = fusedParams
    )

    // Step 5: Rebuild the graph - remove both nodes, add fused node
    val newNodes = graph.nodes.flatMap { node =>
      if (node.id == neuronNode.id || node.id == affineNode.id) {
        None // Remove both nodes
      } else if (nodesAfterNeuron.contains(node)) {
        node.previous = Set(fusedNode)
        Some(node)
      } else {
        Some(node)
      }
    }

    // Step 6: Create updated graph
    val updatedGraph = new NIRGraph(newNodes + fusedNode)

    // Step 7: Recursively replace remaining patterns
    reduce(updatedGraph)
  }

  def fromRaw(rawNodes: Set[RawNode]): NIRGraph = {
    val topRaw = rawNodes
      .find(_.params.nirType == "Input")
      .getOrElse(
        throw new Exception(
          "NIR Graph does not contain node type 'Input', please re-generate."
        )
      )

    val top: NIRNode = NIRNode(topRaw.id, Set.empty[NIRNode], topRaw.params)
    // TODO: put initial case into recursive function
    val nodes = Set(top) ++ convertRawNodes(rawNodes, top)
    new NIRGraph(nodes)
  }

  def apply(f: File): NIRGraph = {
    NIRFileMapper.loadGraph(f)
  }

  private def convertRawNodes(
      nodes: Set[RawNode],
      top: NIRNode
  ): Set[NIRNode] = {
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
        throw new RuntimeException(
          "error: multiple connections not yet supported or no connections found."
        )
    }
  }

  private def getStructureString(node: NIRNode): String = {
    node.params match {
      case i: InputParams => s"${node.id}: $i"
      case x =>
        node.previous.collectFirst { case node: NIRNode =>
          getStructureString(node)
        } + s"${node.id}: $x"
    }
  }
}
