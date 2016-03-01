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
  def splitTrainTest(ratio: Double, data: Iterable[Instance]) :
      (DataSet, DataSet) =
  {
    // Randomize data:
    val shuffled = (new scala.util.Random()).shuffle(data)
    val size = data.size
    val split = (ratio * size).toInt
    (new DataSet(shuffled.slice(0, split).toArray),
      new DataSet(shuffled.slice(split, size).toArray))
  }

  // Compute the average error for a neural network, assuming that all
  // outputs are exclusive binary categories.
  def nnBinaryError(set: DataSet, nn : BackPropagationNetwork) : Double = {
    val size = set.size()
    // Tally up how many outputs are incorrect:
    val incorrect = set.getInstances().map(inst => {
      // Get the actual value:
      val actual = inst.getLabel().getData()
      // Get the predicted value:
      nn.setInputValues(inst.getData())
      nn.run()
      val pred = nn.getOutputValues()
      // Now, these are both binary vectors. To get the actual
      // classification the same way that the R code does, we want the
      // maximum value in each vector.
      if (pred.argMax() == actual.argMax()) 0 else 1
    // TODO (maybe): Make a confusion matrix from these
    }).sum
    // Then, turn this to an average:
    incorrect.toDouble / size
  }

  // Given a training dataset, number of nodes at each layer of the
  // neural network, and a function which produces an
  // OptimizationAlgorithm, construct a neural network and train it
  // one step at a time with the returned OptimizationAlgorithm.  This
  // returns a Stream of neural networks (with the weights updated
  // each time).
  def optimizeNN(
    set: DataSet,
    nodes: Iterable[Int],
    optFn: NeuralNetworkOptimizationProblem => OptimizationAlgorithm) :
      Stream[BackPropagationNetwork] =
  {
    // Build neural network & OptimizationAlgorithm:
    val (net, prob) = getNeuralNet(set, nodes)
    val opt = optFn(prob)

    // At every step:
    def next() : Stream[BackPropagationNetwork] = {
      // Train the network one more step & use the updated weights:
      opt.train()
      net.setWeights(opt.getOptimal().getData())
      // Put that at the head of the (lazy) list:
      net #:: next ()
    }
    next ()
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

    val algos = List(
      ("RHC",  x => new RandomizedHillClimbing(x)),
      ("SA",   x => new SimulatedAnnealing(1e11, 0.95, x)),
      ("GA",   x => new StandardGeneticAlgorithm(500, 250, 40, x))
    ) : List[(String, NeuralNetworkOptimizationProblem => OptimizationAlgorithm)];

    val (faultTrain, faultTest) = splitTrainTest(0.75, faults)
    val faultsErr = (v : (BackPropagationNetwork,Int)) => {
      val (bpn, idx) = v
      val trainErr = nnBinaryError(faultTrain, bpn)
      val testErr = nnBinaryError(faultTest, bpn)
      val trainPct = trainErr * 100.0
      val testPct = testErr * 100.0
      println(f"Iter $idx%d: $trainPct%.2f%% train, $testPct%.2f%% test error")
    }
    for ((name, algo) <- algos) {
      println(s"Faults, $name:")
      // Potential problem here: Do we really want to randomize results each time?
      val nets = optimizeNN(faultTrain, List(27, 25, 7), algo)
      nets.take(10).zipWithIndex.foreach(faultsErr)
    }

    // Temperature value is multiplied by cooling factor at each
    // iteration, that is, the temperature at iteration N is T*cool^N.
    // Thus, to get temperature Tf at iteration N starting from temp
    // T0, Tf = T0*cool^n, Tf/T0 = cool^n, cool = (Tf/T0)^(1/n).
    //val opt = new SimulatedAnnealing(1e11, 0.95, prob)

    //for (w <- net.getWeights()) {
    //  println(w)
    //}

    // Letter recognition is normal CSV; first field is output (it's a
    // character), 16 fields after are inputs:
    println(s"Reading $lettersFile:")
    val letters = CSVReader.open(lettersFile).all().map( row => {
      // Output is a character from A-Z, so we turn it to 26 outputs:
      val out = (0 to 26).map( idx => {
        if ((row(0)(0).toInt - 65) == idx) 1.0 else 0.0
      })
      //println(out)
      instance(row.slice(1, 17).map( x => x.toDouble ), out)
    })
    val lettersRows = letters.size
    println(f"Read $lettersRows%d rows.")

    val algos2 = List(
      ("RHC",  x => new RandomizedHillClimbing(x)),
      ("SA",   x => new SimulatedAnnealing(1e11, 0.95, x)),
      ("GA",   x => new StandardGeneticAlgorithm(500, 250, 40, x))
    ) : List[(String, NeuralNetworkOptimizationProblem => OptimizationAlgorithm)];

    val (letterTrain, letterTest) = splitTrainTest(0.75, letters)
    val lettersErr = (v : (BackPropagationNetwork,Int)) => {
      val (bpn, idx) = v
      val trainErr = nnBinaryError(letterTrain, bpn)
      val testErr = nnBinaryError(letterTest, bpn)
      val trainPct = trainErr * 100.0
      val testPct = testErr * 100.0
      println(f"Iter $idx%d: $trainPct%.2f%% train, $testPct%.2f%% test error")
    }
    
    for ((name, algo) <- algos2) {
      println(s"Letters, $name:")
      // Potential problem here: Do we really want to randomize results each time?
      val nets = optimizeNN(letterTrain, List(16, 8, 26), algo)
      nets.take(10).zipWithIndex.foreach(lettersErr)
    }
  }

}
