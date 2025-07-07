package nir
import nir.NIRNode

case class NIRGraph(nodes: Set[NIRNode]) {
  lazy val top: NIRNode = nodes.find(_.id == "input").get
  lazy val bot: NIRNode = nodes.find(_.id == "output").get
  lazy val nodeMap: Map[String, NIRNode] = nodes.map(node => node.id -> node).toMap

  def getNode(id: String): NIRNode = nodeMap(id)

}

object NIRGraph {
  val empty: NIRGraph = NIRGraph(Set.empty[NIRNode])
  def fromSet(nodes: Set[NIRNode]): NIRGraph = NIRGraph(nodes)
}
