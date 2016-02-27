// Chris Hodapp (chodapp3@gatech.edu)
// Georgia Institute of Technology
// CS7641, Machine Learning, Spring 2016
// Assignment 2, Randomized Optimization (2016-02-22)

import com.github.tototoshi.csv._

import dist.DiscreteDependencyTree
import dist.DiscreteUniformDistribution
import dist.Distribution
import func.nn.backprop._
import opt._
import opt.example._
import opt.ga._
import opt.prob.GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC
import opt.prob.ProbabilisticOptimizationProblem
import shared._
import util.linalg.Vector

object RandomizedOptimization {

  val faultsFile = "Faults.NNA"
  val lettersFile = "letter-recognition.data"

  def tabReader(fname : String) : CSVReader = {
    implicit object TabFormat extends DefaultCSVFormat {
      override val delimiter = '\t'
    }
    return CSVReader.open(fname)
  }

  def main(args: Array[String]) {

    // Faults data is tab-separated:
    println(s"Reading $faultsFile:")
    val faults = tabReader(faultsFile).all().map( row => {
      // 27 float fields & 7 integer (binary) fields:
      val input = row.slice(0, 26).map( x => x.toDouble )
      // This is actually an integer:
      val output = row.slice(27, 33).map( x => x.toDouble )
      val inst = new shared.Instance(input.toArray)
      inst.setLabel(new shared.Instance(output.toArray))
      inst
    })
    val faultRows = faults.size
    println(f"Read $faultRows%d rows.")

    // Letter recognition is normal CSV:
    println(s"Reading $lettersFile:")
    val letters = CSVReader.open(lettersFile).all().map( row => {
      // First field is a single character:
      // val output = row(0)(0);
      // This needs to be a double for an Instance.
      // Next 16 fields are all integers but we need doubles:
      val input = row.slice(1, 16).map( x => x.toDouble );
      val inst = new shared.Instance(input.toArray)
      // inst.setLabel(new shared.Instance(output.toArray))
      inst
    })
    val lettersRows = letters.size
    println(f"Read $lettersRows%d rows.")
  }

}
