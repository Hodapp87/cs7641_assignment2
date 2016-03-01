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

  // Given a dataset, training/test ratio, number of nodes at each
  // layer of the neural network, and a function which produces an
  // OptimizationAlgorithm, construct a neural network and train it
  // one step at a time with the returned OptimizationAlgorithm.  This
  // returns a Stream of (training error, testing error).
  def optimizeNN(
    set: Iterable[Instance], ratio: Double, nodes: Iterable[Int],
    optFn: NeuralNetworkOptimizationProblem => OptimizationAlgorithm) :
      Stream[(Double, Double)] =
  {
    // Separate data, build neural network & OptimizationAlgorithm:
    val (train, test) = splitTrainTest(ratio, set)
    val trainSet = new DataSet(train.toArray)
    val testSet = new DataSet(test.toArray)
    val (net, prob) = getNeuralNet(trainSet, nodes)
    val opt = optFn(prob)

    // At every step...
    def next() : Stream[(Double, Double)] = {
      // Train the network & use the updated weights:
      opt.train()
      net.setWeights(opt.getOptimal().getData())
      // Get training & testing error:
      val trainErr = neuralNetError(trainSet, net)
      val testErr = neuralNetError(testSet, net);
      // Put that at the head of the (lazy) list:
      (trainErr, testErr) #:: next ()
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

    val echoErr = (tup: ((Double, Double), Int)) => {
      val ((trainErr, testErr), idx) = tup
      val trainPct = trainErr * 100.0
      val testPct = testErr * 100.0
      println(f"Iter $idx%d: $trainPct%.2f%% train, $testPct%.2f%% test error")
    }

    val algos = List(
      ("RHC",  x => new RandomizedHillClimbing(x)),
      ("SA",   x => new SimulatedAnnealing(1e11, 0.95, x)),
      ("GA",   x => new StandardGeneticAlgorithm(200, 100, 10, x))
    ) : List[(String, NeuralNetworkOptimizationProblem => OptimizationAlgorithm)]
    for ((name, algo) <- algos) {
      println(s"Faults, $name:")
      // Potential problem here: Do we really want to randomize results each time?
      val training = optimizeNN(faults, 0.75, List(27, 30, 7), algo)
      training.take(10).zipWithIndex.foreach(echoErr)
    }

    // Temperature value is multiplied by cooling factor at each
    // iteration, that is, the temperature at iteration N is T*cool^N.
    // Thus, to get temperature Tf at iteration N starting from temp
    // T0, Tf = T0*cool^n, Tf/T0 = cool^n, cool = (Tf/T0)^(1/n).
    //val opt = new SimulatedAnnealing(1e11, 0.95, prob)

    //for (w <- net.getWeights()) {
      //println(w)
    //}

    // Letter recognition is normal CSV; first field is output (it's a
    // character), 16 fields after are inputs:
    println(s"Reading $lettersFile:")
    val letters = CSVReader.open(lettersFile).all().map( row => {
      instance(row.slice(1, 16).map( x => x.toDouble ), // input
        List(row(0)(0).toInt - 65)) // output (just a char)
    })
    val lettersRows = letters.size
    println(f"Read $lettersRows%d rows.")
    for ((name, algo) <- algos) {
      println(s"Letters, $name:")
      // Potential problem here: Do we really want to randomize results each time?
      val training = optimizeNN(letters, 0.75, List(16, 16, 1), algo)
      training.take(10).zipWithIndex.foreach(echoErr)
    }
  }

}
