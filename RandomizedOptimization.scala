// Chris Hodapp (chodapp3@gatech.edu)
// Georgia Institute of Technology
// CS7641, Machine Learning, Spring 2016
// Assignment 2, Randomized Optimization (2016-03-13)

// Before I forget:
// sbt -Dsbt.log.noformat=true compile

// Java dependencies:
import java.io._
import java.util.Calendar

// CSV & JSON dependencies:
import com.github.tototoshi.csv._
// TODO: Look at https://nrinaudo.github.io/kantan.csv/
import argonaut._, Argonaut._

// ABAGAIL dependencies:
import dist.DiscreteDependencyTree
import dist.DiscreteUniformDistribution
import dist.Distribution
import func.nn.activation._
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

  def main(args: Array[String]) {
    // If 'true', then use the full datasets; if false, then greatly
    // reduce them so that we don't blow up Travis when committing,
    // but still test the same basic code paths.
    val full = false;
    letterRecognition(full)
    steelFaults(full)
  }

    // Run everything for the steel faults classification problem.
  def steelFaults(full : Boolean) {
    // --------------------------------------------------------------------
    // Input & output filenames
    // --------------------------------------------------------------------
    val faultsFile = "Faults.NNA"
    //val faultsOutput = "faults-nn9.json"
    val faultsOutput = "faults-nn-dummy.json"
    
    // --------------------------------------------------------------------
    // Data loading and conditioning
    // --------------------------------------------------------------------
    // Faults data is tab-separated with 27 inputs, 7 outputs:
    println(s"Reading $faultsFile:")
    val faultsRaw = tabReader(faultsFile).all().map( row => {
      (row.slice(0, 27).map( x => x.toDouble ), // input
        row.slice(27, 34).map( x => x.toDouble )) // output
    })
    val faultsCond = conditionAttribs(faultsRaw)
    val faults = if (full) faultsCond else
      faultsCond.take((0.05 * faultsCond.size).toInt)
    val faultRows = faults.size
    // TODO: Factor the above code into a function if possible.
    println(f"Read $faultRows%d rows.")

    // --------------------------------------------------------------------
    // A note on simulated annealing:
    // Temperature value is multiplied by cooling factor at each
    // iteration, that is, the temperature at iteration N is T*cool^N.
    // Thus, to get temperature Tf at iteration N starting from temp
    // T0, Tf = T0*cool^n, Tf/T0 = cool^n, cool = (Tf/T0)^(1/n).
    // --------------------------------------------------------------------

    val algos = List(
      /*("RHC", x => new RandomizedHillClimbing(x)),
      ("SA, 1e11 & 0.90", x => new SimulatedAnnealing(1e11, 0.90, x)),
      ("SA, 1e11 & 0.95", x => new SimulatedAnnealing(1e11, 0.95, x)),
      ("SA, 1e10 & 0.95", x => new SimulatedAnnealing(1e10, 0.95, x)),
      ("SA, 1e12 & 0.90", x => new SimulatedAnnealing(1e11, 0.95, x)),*/
      ("GA, 125, 88, 38", x => new StandardGeneticAlgorithm(125, 88, 38, x)),
      ("GA, 250, 175, 75", x => new StandardGeneticAlgorithm(250, 175, 75, x)),
      ("GA, 500, 350, 150", x => new StandardGeneticAlgorithm(500, 350, 150, x)),
      ("GA, 125, 113, 38", x => new StandardGeneticAlgorithm(125, 113, 38, x)),
      ("GA, 250, 225, 75", x => new StandardGeneticAlgorithm(250, 225, 75, x)),
      ("GA, 500, 450, 150", x => new StandardGeneticAlgorithm(500, 450, 150, x))
    ) : List[(String, NeuralNetworkOptimizationProblem => OptimizationAlgorithm)];

    val split = 0.75
    val (train, test) = splitTrainTest(split, faults)
    val trainSize = train.size
    val testSize = test.size
    val iters = if (full) 10000 else 500
    val runs = (1 to 1)
    val nodeList = List(20)
    // Kludge alert:
    var weights = scala.collection.mutable.ListBuffer.empty[TrainWeights]
    // We append to this in the loop below because I don't really know
    // of another good way to do this.
    println(s"Training: $trainSize, testing: $testSize")
    val results = algos.par.flatMap { case (name,algo) =>
      runs.par.flatMap { run =>
        nodeList.par.flatMap { hiddenNodes =>
          println(s"Starting $name, run $run, $hiddenNodes nodes")
          val nets = optimizeNN(train, List(27, hiddenNodes, 7), algo)
          nets.zipWithIndex.take(iters).flatMap { case (bpn,iter) =>
            val ctxt = TestParams("faults", name, split, hiddenNodes, run, iter)
            val trainErr = nnBinaryError(train, bpn)
            val testErr = nnBinaryError(test, bpn)
            val testRec = ErrorResult(ctxt, trainErr, testErr)
            if (iter % 100 == 0)
              println(testRec)
            if (iter == (iters - 1)) {
              val w = bpn.getWeights
              val wsize = w.size
              println(s"Saving $wsize weights...")
              weights += TrainWeights(ctxt, w)
            }
            List(testRec)
          }
        }
      }
    }

    {
      val algoList = algos.map(_._1)
      val date = Calendar.getInstance().getTime()
      val testId = f"Steel faults classification, started on $date, algorithms: $algoList, $split%.3f split, $iters iterations, hidden nodes tested: $nodeList"
      val writer = new PrintWriter(new File(faultsOutput))
      writeJson(writer, testId, weights, results)
      writer.close()
    }

    //for (w <- net.getWeights()) {
    //  println(w)
    //}
  }

  // Run everything for the steel letter recognition problem.
  def letterRecognition(full : Boolean) {
    val lettersFile = "letter-recognition.data"
    //val lettersOutput = "letters-nn6.json"
    val lettersOutput = "letters-nn-dummy.json"

    // Letter recognition is normal CSV; first field is output (it's a
    // character), 16 fields after are inputs:
    println(s"Reading $lettersFile:")
    val lettersRaw = CSVReader.open(lettersFile).all().map( row => {
      // Output is a character from A-Z, so we turn it to 26 outputs:
      val letterClass = { (idx : Int) =>
        if ((row(0)(0).toInt - 65) == idx) 1.0 else 0.0
      }
      (row.slice(1, 17).map( x => x.toDouble ), // input
        (0 to 26).map(letterClass)) // output
    })
    val lettersCond = conditionAttribs(lettersRaw)
    val letters = if (full) lettersCond else
      lettersCond.take((0.05 * lettersCond.size).toInt)

    val lettersRows = letters.size
    println(f"Read $lettersRows%d rows.")

    val algos = List(
      //("RHC", x => new RandomizedHillClimbing(x)),
      ("SA, 1e12 & 0.99", x => new SimulatedAnnealing(1e12, 0.99, x)),
      ("SA, 1e10 & 0.99", x => new SimulatedAnnealing(1e10, 0.99, x)),
      ("SA, 1e8 & 0.99", x => new SimulatedAnnealing(1e8, 0.99, x)),
      ("SA, 1e12 & 0.98", x => new SimulatedAnnealing(1e12, 0.98, x)),
      ("SA, 1e10 & 0.98", x => new SimulatedAnnealing(1e10, 0.98, x)),
      ("SA, 1e8 & 0.98", x => new SimulatedAnnealing(1e8, 0.98, x))
      //("SA, 1e11 & 0.90", x => new SimulatedAnnealing(1e11, 0.90, x)),
      //("SA, 1e10 & 0.95", x => new SimulatedAnnealing(1e10, 0.95, x))
      //("SA, 1e10 & 0.90", x => new SimulatedAnnealing(1e10, 0.90, x))
    ) : List[(String, NeuralNetworkOptimizationProblem => OptimizationAlgorithm)];

    val split = 0.75
    val (train, test) = splitTrainTest(split, letters)
    val iters = if (full) 100000 else 200
    val runs = (1 to 1)
    val nodeList = List(60)
    // Kludge alert:
    var weights = scala.collection.mutable.ListBuffer.empty[TrainWeights]
    // We append to this in the loop below because I don't really know
    // of another good way to do this.
    val results = algos.par.flatMap { case (name,algo) =>
      runs.par.flatMap { run =>
        nodeList.par.flatMap { hiddenNodes =>
          println(s"Starting $name, run $run, $hiddenNodes nodes")
          val nets = optimizeNN(train, List(16, hiddenNodes, 26), algo)
          nets.zipWithIndex.take(iters).flatMap { case (bpn,iter) =>
            val ctxt = TestParams("letters", name, split, hiddenNodes, run, iter)
            val trainErr = nnBinaryError(train, bpn)
            val testErr = nnBinaryError(test, bpn)
            val testRec = ErrorResult(ctxt, trainErr, testErr)
            if (iter % 100 == 0)
              println(testRec)
            if (iter == (iters - 1)) {
              val w = bpn.getWeights
              val wsize = w.size
              println(s"Saving $wsize weights...")
              weights += TrainWeights(ctxt, w)
            }
            List(testRec)
          }
        }
      }
    }

    {
      val algoList = algos.map(_._1)
      val date = Calendar.getInstance().getTime()
      val testId = f"Letters classification, started on $date, algorithms: $algoList, $split%.3f split, $iters iterations, hidden nodes tested: $nodeList"
      val writer = new PrintWriter(new File(lettersOutput))
      writeJson(writer, testId, weights, results)
      writer.close()
    }
  }

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
    val net = factory.createClassificationNetwork(nodes.toArray, new LogisticSigmoid())
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

  // Class giving the parameters/context for some piece of information
  // on the test
  case class TestParams(test: String, name: String, split: Double,
    hiddenNodes: Int, run: Int, iter: Int)

  // Class giving the training & testing error in some context
  case class ErrorResult(params: TestParams, trainErr: Double, testErr: Double)

  implicit def ErrorResultJson: EncodeJson[ErrorResult] =
    EncodeJson((p: ErrorResult) => {
      val ctxt = p.params
      ("test"        := jString(ctxt.test))        ->:
      ("name"        := jString(ctxt.name))        ->:
      ("run"         := jNumber(ctxt.run))         ->:
      ("split"       := jNumber(ctxt.split))       ->:
      ("hiddenNodes" := jNumber(ctxt.hiddenNodes)) ->:
      ("iter"        := jNumber(ctxt.iter))        ->:
      ("trainErr"    := jNumber(p.trainErr))       ->:
      ("testErr"     := jNumber(p.testErr))        ->:
      jEmptyObject
    })

  // Class giving neural network's weights at some stage of training
  case class TrainWeights(params: TestParams, weights: Array[Double])

  implicit def TrainWeightsJson: EncodeJson[TrainWeights] =
    EncodeJson((p: TrainWeights) => {
      val ctxt = p.params
      val weights = p.weights.map(w => jNumber(w).getOrElse(jNumber(0))).toList
      ("test"        := jString(ctxt.test))        ->:
      ("name"        := jString(ctxt.name))        ->:
      ("run"         := jNumber(ctxt.run))         ->:
      ("split"       := jNumber(ctxt.split))       ->:
      ("hiddenNodes" := jNumber(ctxt.hiddenNodes)) ->:
      ("iter"        := jNumber(ctxt.iter))        ->:
      ("weights"     := jArray(weights))           ->:
      jEmptyObject
    })

  // Write the given records to the Writer (which will not be closed).
  // TODO: Make all of this more general.  I'm using 'for' (via
  // 'flatMap', I think), 'take', and 'drop' on the ParSeq, and there
  // is a more general trait I can use, but I don't know how.
  def writeJson(f: Writer, testId: String,
    weights: Iterable[TrainWeights],
    errors: scala.collection.parallel.immutable.ParSeq[ErrorResult])
  {
    // First, write out the test ID:
    f.write("{")
    // I am open to suggestions for how to do this right.  The problem
    // is that if I make it something like (("testId" :=
    // jString(testId)) ->: jEmptyObject) then it's wrapped in another
    // object, and jsonlite doesn't want to read it in that way.
    f.write(s""""testId": "$testId"""")
    f.flush()
    // This is sort of manually writing the array because of
    // https://github.com/argonaut-io/argonaut/issues/52
    f.write(",\"data\": [")
    // The first record must be written with no leading commo:
    f.write(errors.apply(0).asJson.spaces2)
    // ...so that the rest can all be written with a leading comma,
    // and then regardless of the order, come out fine:
    for (r <- errors.drop(1)) {
      f.write("," + r.asJson.spaces2)
      f.flush()
    }
    f.write("],\"weights\": ")
    f.write(jArray(weights.map(w => w.asJson).toList).spaces2)
    f.write("}")
  }

  // Normalize the given data to have mean of 0 and variance of 1.
  def normalize(data: Iterable[Double]) : Iterable[Double] = {
    val mean = data.sum / data.size
    val meanZero = data.map( d => d - mean )
    val variance = meanZero.map( d => d*d ).sum / data.size
    meanZero.map( d => d / Math.sqrt(variance) )
  }

  // Turn an Iterable of (input, output) instances into an iterable of
  // ABAGAIL Instance objects, containing each input, normalized and
  // labeled with corresponding output.
  // 
  // More specifically: Given an Iterable of (input, output) where
  // each element corresponds to a single row (or instance) of data,
  // each one consisting of an Iterable of inputs and an Iterable of
  // outputs (that is, a value in every attribute of input and
  // output), return an Iterable of Instances which contain the input
  // values, normalized on each attribute, and labeled with the
  // outputs.  Outputs are assumed to require no normalization.
  def conditionAttribs(in: Iterable[(Iterable[Double], Iterable[Double])]) :
      Iterable[Instance] =
  {
    // 'in' contains a list of tuples, one tuple per *row* of data.
    // 'inTrans' separates this out into a list of lists and then
    // transposes it - so each outer list contains one entire
    // *attribute* of the input:
    val inTrans = in.map(t => t._1).transpose
    // We then normalize each attribute and transpose back, so
    // 'inNormed' then contains a list of lists, each outer list
    // containing one *row* of input:
    val inNormed = inTrans.map(normalize).transpose
    // We don't need to transform the output at all, so simply extract
    // it from the tuple and then this is the same format as
    // 'inNormed':
    val out = in.map { t => t._2 }
    // Finally, turn these into a form ABAGAIL will take:
    inNormed.zip(out).map { case (in, out) => instance(in, out) }
  }

}
