# Installation
Currently, the only supported way is by using the repository as a submodule, and pointing **sbt** to it.

1. Download the github repository into your sbt project, *proj*.
   ```bash
   cd proj
   git clone https://github.com/mrontio/nir4s.git
   ```
2. In your `build.sbt`, instantiate the project as the `nir` package.
   ```scala
   lazy val nir = ProjectRef(file("./nir4s"), "nir")
   ```
3. Use it inside of your Scala files.
   ```scala
   import nir._
   import tensors._
   ```
