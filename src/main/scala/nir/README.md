# Generator

-   The generator takes a NIR .nir HDF5 file, parses it, and gets it into a Scala structure that can then easily be used to generate hardware.

## Checklist

-   [X] Define NIR Nodes in Scala
    -   Class [NIRNodes](NIRNode.scala) using the trait system
    -   Each node has 'previous', not next
-   [-] Define NIR Graph in Scala
    -   [NIRGraph](NIRGraph.scala) represents this
    -   [X] Initialize from [Output](NIRGraph.scala), so we can build it back up
    -   [X] Initialise Empty, this is a better approach because we can recurse down.
    -   [ ] [Traversal operator](NIRGraph.scala)
-   [X] Read NIR into Scala: [NIRMapper](NIRMapper.scala)
-   [ ] Convert to hardware
    -   Consult Francisco with the final version
    -   We have agreed on the current structure, now just need to plug it into his generator.
