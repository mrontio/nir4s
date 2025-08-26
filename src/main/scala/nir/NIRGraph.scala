package nir

case class NIRGraph(nodes: Set[NIRNode]) {
  val input: NIRNode = nodes.find(_.previous.size == 0).get
  val output: NIRNode = nodes.find(_.id == "output").get
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

  def fromRaw(rawNodes: Set[RawNode]): NIRGraph = {
    val topRaw: RawNode = rawNodes.find(_.id == "input").get
    val top: NIRNode = NIRNode(topRaw.id, Set.empty[NIRNode], topRaw.params)
    // TODO: put initial case into recursive function
    val nodes = Set(top) ++ convertRawNodes(rawNodes, top)
    NIRGraph(nodes)
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
