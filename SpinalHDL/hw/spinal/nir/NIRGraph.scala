package nir
import nir.NIRNode

case class NIRGraph(nodes: Set[NIRNode]) {
  val top: NIRNode = nodes.find(_.id == "input").get
  // lazy val bot: NIRNode = nodes.find(_.id == "output").get
  val nodeMap: Map[String, NIRNode] = nodes.map(node => node.id -> node).toMap

  def getNode(id: String): NIRNode = nodeMap(id)
}

object NIRGraph {
  def fromRaw(rawNodes: Set[RawNode]): NIRGraph = {
    val rawNodeMap: Map[String, RawNode] = rawNodes.map(node => node.id -> node).toMap
    val topRaw: RawNode = rawNodes.find(_.id == "input").get
    val top: NIRNode = NIRNode(topRaw.id, Set.empty[NIRNode], topRaw.params)
    graphFromRaw(rawNodeMap, top)
  }

  private def graphFromRaw(map: Map[String, RawNode], top: NIRNode): NIRGraph = {
    // find previous link that goes to top
    val previous: Iterable[RawNode] = map.values.collect {
      case n if n.prevIds.contains(top.id) => n
    }
    NIRGraph(Set[NIRNode](top))

    // construct them one by one
    // set both of them to top
    // base case: nothing links to top
    // base case: the node we link to links to use (recurrence)
    // recursion case: graphFromRaw(map, linked_to_top)
  }
}
