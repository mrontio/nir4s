# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NIR4S is a Scala library for parsing and manipulating Neuromorphic Intermediate Representation (NIR) files. It converts HDF5-based NIR files into Scala ADT representations for neuromorphic computing research with spiking neural networks.

**Repository**: https://github.com/mrontio/nir4s
**Documentation**: https://mront.io/nir4s/
**License**: MIT (2025)

## Build Commands

### Testing
```bash
# Run all tests
sbt test

# Run specific test suite
sbt "testOnly nir.TensorDynamicSpec"
sbt "testOnly nir.TensorStaticSpec"
sbt "testOnly nir.NIRSpec"

# Run tests in watch mode
sbt ~test
```

### Building
```bash
# Compile the project
sbt compile

# Generate Scaladoc API documentation
sbt doc
# Output: target/scala-2.13/api/

# Clean build artifacts
sbt clean
```

### Documentation
```bash
# Build MkDocs documentation site
mkdocs build
# Output: site/

# Serve documentation locally
mkdocs serve
# Available at http://127.0.0.1:8000

# Generate Scaladoc and copy to docs/api for MkDocs integration
sbt doc
cp -r target/scala-2.13/api/ docs/api/
```

### Code Formatting
```bash
# Format all Scala code
sbt scalafmt

# Check formatting without changes
sbt scalafmtCheck
```

## Architecture

### Core Components

**NIRGraph** (`NIRGraph.scala`)
Directed acyclic graph representation with node connectivity resolution. Key methods:
- `fromRaw(rawNodes: Set[RawNode])` - Converts string-based node references to pointer-based graph
- `reduceConv2DIFSubgraph(graph: NIRGraph)` - Fuses Conv2D + IF neuron patterns into Conv2DIF nodes (recursive)
- `apply(f: File)` - Loads graph from HDF5 file via NIRFileMapper

**NIRNode** (`NIRNode.scala`)
Two-stage node representation:
- `RawNode` - Unresolved nodes with string-based predecessor IDs (from HDF5)
- `NIRNode` - Resolved nodes with actual predecessor references
- `NIRParams` - Sealed trait with 13+ case classes for operation types

**Tensor System** (`tensor/`)
Dual tensor representation pattern:
- `TensorDynamic` - Runtime shape checking, mutable indexing via `Indexer`, supports 1D-8D
- `TensorStatic` - Compile-time type safety with `Tensor1D`/`Tensor2D`/`Tensor3D`/`Tensor4D` variants
- Conversion: `TensorDynamic.toStatic` for upgrading to static types
- `Iso.scala` - RangeTree structure for efficient multidimensional indexing (row-major order)

**NIRFileMapper** (`NIRFileMapper.scala`)
HDF5 file parsing and NIR graph construction. Converts HDF5 datasets to TensorDynamic instances.

### Supported NIR Node Types

**Neurons**: LI, IF, I, LIF, CubaLIF
**Layers**: Conv1d, Conv2d, Conv2dIF (fused), Linear, Affine, SumPool2d, Flatten
**I/O**: Input, Output

### Graph Resolution Pattern

1. Load HDF5 → `Set[RawNode]` (string-based predecessor IDs)
2. `NIRGraph.fromRaw()` → `Set[NIRNode]` (pointer-based predecessors)
3. Optional: `reduceConv2DIFSubgraph()` → Fused subgraph nodes

### Key Design Notes

- **Mutable state**: `NIRNode.previous` is `var` (marked TODO for refactoring)
- **Subgraph reduction**: Recursive pattern matching to fuse Conv2D + IF neurons
- **Single-chain assumption**: Graph conversion currently supports linear chains (multiple connections throw RuntimeException at NIRGraph.scala:117)
- **Test data**: `src/test/scala/nir/samples/` contains reference NIR files (conv1, i, li, lif, cnn_sinabs)

## Dependencies

- **Scala**: 2.13.14
- **HDF5**: io.jhdf 0.6.5
- **JSON**: circe 0.14.9 (core, generic, parser)
- **FP**: cats-core 2.10.0, shapeless 2.3.10
- **Testing**: munit 0.7.29
- **Logging**: slf4j 2.0.13

## Project Structure

```
src/main/scala/nir/
├── NIRGraph.scala          # Graph structure, conversion, transformations
├── NIRNode.scala           # Node definitions (RawNode, NIRNode, NIRParams hierarchy)
├── NIRFileMapper.scala     # HDF5 parsing
├── package.scala           # Matrix type aliases
├── Main.scala              # Entry point
└── tensor/
    ├── TensorDynamic.scala # Runtime-flexible tensors
    ├── TensorStatic.scala  # Compile-time safe tensors
    └── Iso.scala           # RangeTree indexing structure

src/test/scala/nir/
├── TensorDynamicSpec.scala # Tensor operations tests
├── TensorStaticSpec.scala  # Static tensor type tests
├── NIRSpec.scala           # Graph loading tests
├── DocSpec.scala           # Documentation tests
└── samples/                # Test NIR files (conv1, i, li, lif, cnn_sinabs)
```

## Common Development Patterns

### Adding New NIR Node Types

1. Add case class extending `NIRParams` in `NIRNode.scala`
2. Implement `nirType: String` and `toString: String`
3. Update `NIRFileMapper` to parse from HDF5 format
4. Add test sample in `src/test/scala/nir/samples/`

### Working with Tensors

- Use `TensorDynamic` for I/O operations (HDF5, NumPy, JSON)
- Convert to `TensorStatic` variants for type-safe processing
- Shape validation happens at runtime for TensorDynamic, compile-time for TensorStatic
- Indexing uses row-major (C-style) ordering via RangeTree

### Graph Transformations

Follow the pattern in `reduceConv2DIFSubgraph()`:
1. Filter nodes by pattern
2. Extract and transform parameters
3. Create fused node
4. Rebuild graph with updated node references
5. Recurse if needed
