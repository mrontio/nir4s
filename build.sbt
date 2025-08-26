import Dependencies._

ThisBuild / scalaVersion     := "2.13.14"

val javaHDF = "io.jhdf" % "jhdf" % "0.6.5"
val munitTest = munit % Test
val slf4j_api = "org.slf4j" % "slf4j-api" % "2.0.13"
val slf4j_console = "org.slf4j" % "slf4j-simple" % "2.0.13"
val shapeless = "com.chuusai" %% "shapeless" % "2.3.10"



lazy val nir = (project in file("."))
  .settings(
    name := "nir",
    scalaVersion     := "2.13.14",
    libraryDependencies ++= Seq(slf4j_api, slf4j_console, munitTest, javaHDF, shapeless)
  )

// See https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html for instructions on how to publish to Sonatype.
