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

  // Produce a neural net (and an OptimizationProblem) given data and
  // the number of nodes at each layer.
  def getNeuralNet(set: DataSet, nodes: Iterable[Int]) :
      (BackPropagationNetwork, NeuralNetworkOptimizationProblem) =
  {
    val factory = new BackPropagationNetworkFactory()
    val net = factory.createClassificationNetwork(nodes.toArray)
    val sse = new SumOfSquaresError()
    val opt = new NeuralNetworkOptimizationProblem(set, net, sse)
    (net, opt)
  }

  // Split the given data randomly into (training, testing) where
  // 'ratio' gives the amount to devote to each (e.g. ratio = 0.75 is
  // 75% for training, 25% for testing).
  def splitTrainTest(ratio: Float, data: Iterable[Instance]) :
      (Iterable[Instance], Iterable[Instance]) =
  {
    // Randomize data:
    val shuffled = (new scala.util.Random()).shuffle(data)
    val size = data.size
    val split = (ratio * size).toInt
    (shuffled.slice(0, split), shuffled.slice(split, size))
  }

  // Compute the error 
  def neuralNetError(set: DataSet, nn : BackPropagationNetwork) : Double = {
    val size = set.size()
    // Tally up how many outputs are incorrect:
    val incorrect = set.getInstances().map(inst => {
      nn.setInputValues(inst.getData())
      nn.run()
      val pred = inst.getLabel().getData()
      val actual = nn.getOutputValues()
      val err = pred.minus(actual).normSquared()
      if (err >= 1) 1 else 0
    }).sum
    incorrect.toDouble / size
  }

  def main(args: Array[String]) {

    // Faults data is tab-separated with 27 inputs, 7 outputs:
    println(s"Reading $faultsFile:")
    val faults = tabReader(faultsFile).all().map( row => {
      instance(row.slice(0, 27).map( x => x.toDouble ), // input
        row.slice(27, 34).map( x => x.toDouble )) // output
    })
    val (faultTrain, faultTest) = splitTrainTest(0.8f, faults);
    {
      val faultRows = faults.size
      println(f"Read $faultRows%d rows.")
      val (trainSize, testSize) = (faultTrain.size, faultTest.size)
      println(f"Split $trainSize%d for training, $testSize%d for testing.")
    }
    val faultSetTrain = new DataSet(faultTrain.toArray)
    val faultSetTest = new DataSet(faultTest.toArray)
    // Problem(ish): I need 'prob' to train, and 'net' to test later on.
    val (net, prob) = getNeuralNet(faultSetTrain, List(27, 30, 7))

    //val rhc = new RandomizedHillClimbing(prob)

    // Temperature value is multiplied by cooling factor at each
    // iteration, that is, the temperature at iteration N is T*cool^N.
    // Thus, to get temperature Tf at iteration N starting from temp
    // T0, Tf = T0*cool^n, Tf/T0 = cool^n, cool = (Tf/T0)^(1/n).
    val sa = new SimulatedAnnealing(1e11, 0.95, prob)
    for (i <- 1 to 1000) {
      // Iterate (mutating everything):
      sa.train()
      val opt = sa.getOptimal()
      net.setWeights(opt.getData())

      val trainErr = neuralNetError(faultSetTrain, net)
      val testErr = neuralNetError(faultSetTest, net);

      {
        val trainPct = trainErr * 100.0
        val testPct = testErr * 100.0
        println(f"Iter $i%d: $trainPct%.2f%% train, $testPct%.2f%% test error")
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
