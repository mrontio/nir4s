package nir
import nir.NIRNode

case class NIRGraph(nodes: Set[NIRNode]) {
  val input: NIRNode = nodes.find(_.previous.size == 0).get
  val output: NIRNode = nodes.find(_.id == "output").get
  val nodeMap: Map[String, NIRNode] = nodes.map(node => node.id -> node).toMap


  def getNode(id: String): NIRNode = nodeMap(id)
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
    // find previous link that goes to top
    val next: Set[RawNode] = nodes.collect {
      case n if n.prevIds.contains(top.id) => n
    }

    next match {
      case s if s.size == 1 && s.head.id == "output" => {
        // Base case: output
        val newNode = NIRNode(s.head.id, Set(top), s.head.params)
        Set(newNode)
      }
      case s if s.size == 1 && s.head.id == top.id => {
        // Base case: recurrence
        Set(top)
      }
      case s if s.size == 1 => {
        // Recursive case: Single next
        val head = s.head
        val newNode = NIRNode(head.id, Set(top), head.params)
        convertRawNodes(nodes, newNode) ++ Set(newNode)
      }
      case _ => {
        throw new RuntimeException("error: multiple connections not yet supported.")
      }
    }
  }
}
