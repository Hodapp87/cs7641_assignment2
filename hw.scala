// Chris Hodapp (chodapp3@gatech.edu)
// Georgia Institute of Technology
// CS7641, Machine Learning, Spring 2016
// Assignment 2, Randomized Optimization (2016-02-22)

// Before I forget:
// sbt -Dsbt.log.noformat=true run

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

  // Override implicit object CSVReader.open uses & change delimiter:
  def tabReader(fname : String) : CSVReader = {
    implicit object TabFormat extends DefaultCSVFormat {
      override val delimiter = '\t'
    }
    return CSVReader.open(fname)
  }

  // Produce a 'shared.Instance' for ABAGAIL given inputs and outputs.
  def instance(in: Iterable[Double], out: Iterable[Double]) : Instance = {
    val inst = new Instance(in.toArray)
    inst.setLabel(new Instance(out.toArray))
    inst
  }

  // Produce a neural net
  def getNeuralNet(set: DataSet, layers: Iterable[Int]) :
      (BackPropagationNetwork, NeuralNetworkOptimizationProblem) =
  {
    val factory = new BackPropagationNetworkFactory()
    val net = factory.createClassificationNetwork(layers.toArray)
    val sse = new SumOfSquaresError()
    val opt = new NeuralNetworkOptimizationProblem(set, net, sse)
    (net, opt)
  }

  def main(args: Array[String]) {

    // Faults data is tab-separated with 27 inputs, 7 outputs:
    println(s"Reading $faultsFile:")
    val faults = tabReader(faultsFile).all().map( row => {
      instance(row.slice(0, 27).map( x => x.toDouble ), // input
        row.slice(27, 34).map( x => x.toDouble )) // output
    })
    val faultRows = faults.size
    println(f"Read $faultRows%d rows.")
    val faultSet = new DataSet(faults.toArray)
    // Problem(ish): I need 'prob' to train, and 'net' to test later on.
    val (net, prob) = getNeuralNet(faultSet, List(27, 10, 7))

    //val rhc = new RandomizedHillClimbing(prob)
    val sa = new SimulatedAnnealing(1e11, 0.975, prob)
    var errBest = 100.0
    for (i <- 1 to 500) {
      // Iterate (mutating everything):
      sa.train()
      val opt = sa.getOptimal()
      net.setWeights(opt.getData())

      // Yeah yeah yeah... 
      val testSet = faults

      val total = testSet.size
      // Compute training error:
      val err = testSet.map(inst => {
        net.setInputValues(inst.getData())
        net.run()
        val pred = inst.getLabel().getData()
        val actual = net.getOutputValues()
        val err = pred.minus(actual).normSquared()
        if (err >= 1) 1 else 0
      }).sum
      val pct = 100.0 * err / total
      if (pct < errBest) {
        errBest = pct
        println(f"Iteration $i%d: $pct%.2f%% ($err%d incorrect of $total%d)")
      }
    }

    for (w <- net.getWeights()) {
      println(w)
    }

    // Need to call 'train' on the OptimizationAlgorithm.  After each
    // call, I need to set samples at the network's input, get the
    // outputs, and check them for correctness.
    // This repeats for whatever number of iterations is desired.

    //val faultProblems = List(
    //  new RandomizedHillClimbing(getNeuralNet(set, layers
    //)

    // Letter recognition is normal CSV; first field is output (it's a
    // character), 16 fields after are inputs:
    println(s"Reading $lettersFile:")
    val letters = CSVReader.open(lettersFile).all().map( row => {
      instance(row.slice(1, 16).map( x => x.toDouble ), // input
        List(0.0)) // output
    })
    val lettersRows = letters.size
    println(f"Read $lettersRows%d rows.")
    val letterSet = new DataSet(letters.toArray)
  }

}
