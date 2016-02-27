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

  def tabReader(fname : String) : CSVReader = {
    implicit object TabFormat extends DefaultCSVFormat {
      override val delimiter = '\t'
    }
    return CSVReader.open(fname)
  }

  def main(args: Array[String]) {

    // Faults data is tab-separated:
    println("Faults:")
    for (row <- tabReader("Faults.NNA")) {
      // 27 float fields & 7 integer (binary) fields:
      val input = row.slice(0, 26).map( x => x.toDouble );
      val output = row.slice(27, 33).map( x => x.toInt );

      println (input)
      println (output)
    }

    // Letter recognition is normal CSV:
    println("letter-recognition:")
    for (row <- CSVReader.open("letter-recognition.data")) {
      // First field is a single character:
      val output = row(0)(0);
      // Next 16 fields are all integers:
      val input = row.slice(1, 16).map( x => x.toInt );

      println (input)
      println (output)
    }
  }

}
