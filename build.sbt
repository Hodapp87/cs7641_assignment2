lazy val root = (project in file(".")).
  settings(
    name := "chodapp3_assignment2"
  )

libraryDependencies += "com.github.tototoshi" %% "scala-csv" % "1.3.0"

libraryDependencies += "io.argonaut" %% "argonaut" % "6.1"
