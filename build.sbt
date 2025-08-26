import Dependencies._

ThisBuild / scalaVersion     := "2.13.14"

lazy val root = (project in file("."))
  .settings(
    name := "nir",
    libraryDependencies += munit % Test
  )

// See https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html for instructions on how to publish to Sonatype.
