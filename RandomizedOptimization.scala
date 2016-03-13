// Chris Hodapp (chodapp3@gatech.edu)
// Georgia Institute of Technology
// CS7641, Machine Learning, Spring 2016
// Assignment 2, Randomized Optimization (2016-03-13)

// To build without formatted logs (for piping or M-x compile):
// sbt -Dsbt.log.noformat=true compile

// The functionality for counting function calls only works when
// invoked with: -Dscala.concurrent.context.numThreads=1
// -Dscala.concurrent.context.maxThreads=1

// Java dependencies:
import java.io._
import java.util.Calendar

// CSV & JSON dependencies:
import com.github.tototoshi.csv._
// TODO: Look at https://nrinaudo.github.io/kantan.csv/
import argonaut._, Argonaut._

// ABAGAIL dependencies:
import dist.{DiscreteDependencyTree, DiscretePermutationDistribution,
  DiscreteUniformDistribution, Distribution}
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

// Scala dependencies:
import scala.concurrent._
import scala.concurrent.duration._
import ExecutionContext.Implicits.global

object RandomizedOptimization {

  def main(args: Array[String])
  {
    // If 'true', then use the full datasets; if false, then greatly
    // reduce them so that we don't blow up Travis when committing,
    // but still test the same basic code paths.
    val full = true

    //steelFaults(full)
    //letterRecognition(full)
    //knapsackTest()
    //travelingSalesmanTest()
    //continuousPeaksTest()
  }

    // Run everything for the steel faults classification problem.
  def steelFaults(full : Boolean)
  {
    // --------------------------------------------------------------------
    // Input & output filenames
    // --------------------------------------------------------------------
    val faultsFile = "Faults.NNA"
    
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

    {
      val algos = List(
        ("RHC", x => new RandomizedHillClimbing(x)),
        ("SA, 1e2 & 0.9999", x => new SimulatedAnnealing(1e2, 0.9999, x)),
        ("SA, 1e2 & 0.9995", x => new SimulatedAnnealing(1e2, 0.9995, x)),
        ("SA, 1e2 & 0.999", x => new SimulatedAnnealing(1e2, 0.999, x)),
        ("SA, 1e3 & 0.9999", x => new SimulatedAnnealing(1e3, 0.9999, x)),
        ("SA, 1e3 & 0.9995", x => new SimulatedAnnealing(1e3, 0.9995, x)),
        ("SA, 1e3 & 0.999", x => new SimulatedAnnealing(1e3, 0.999, x)),
        ("SA, 1e4 & 0.9999", x => new SimulatedAnnealing(1e4, 0.9999, x)),
        ("SA, 1e4 & 0.9995", x => new SimulatedAnnealing(1e4, 0.9995, x)),
        ("SA, 1e4 & 0.999", x => new SimulatedAnnealing(1e4, 0.999, x))
      ) : List[(String, NeuralNetworkOptimizationProblem => OptimizationAlgorithm)];

      val split = 0.75
      val iters = if (full) 150000 else 500
      val runs = 8
      val nodeList = List(20)
      runNeuralNetTestMatrix("faults", "faults-sa-tmp.json", split, nodeList,
        runs, iters, faults, algos)
    }
    
    // This is to get a good measurement of stdev/min/max for many RHC
    // runs:
    {
      val algos = List(
        ("RHC", x => new RandomizedHillClimbing(x)),
        ("SA, 1e10 & 0.95", x => new SimulatedAnnealing(1e11, 0.99, x))
      ) : List[(String, NeuralNetworkOptimizationProblem => OptimizationAlgorithm)];

      val split = 0.75
      val iters = if (full) 20000 else 500
      val runs = 200
      val nodeList = List(20)
      runNeuralNetTestMatrix("faults", "faults-rhc-200runs.json", split, nodeList,
        runs, iters, faults, algos)
    }

    // This is just to establish timing for RHC & SA:
    {
      val algos = List(
        ("RHC", x => new RandomizedHillClimbing(x)),
        ("SA, 1e11 & 0.99", x => new SimulatedAnnealing(1e11, 0.99, x))
      ) : List[(String, NeuralNetworkOptimizationProblem => OptimizationAlgorithm)];

      val split = 0.75
      val iters = if (full) 10000 else 500
      val runs = 1
      val nodeList = List(10, 20, 30)
      runNeuralNetTestMatrix("faults", "faults-timing.json", split, nodeList,
        runs, iters, faults, algos)
    }

    // This will take awhile to run:
    {
      val algos = List(
        ("RHC", x => new RandomizedHillClimbing(x)),
        ("SA, 1e11 & 0.99", x => new SimulatedAnnealing(1e11, 0.99, x)),
        ("SA, 1e10 & 0.99", x => new SimulatedAnnealing(1e10, 0.99, x)),
        ("SA, 1e9 & 0.99", x => new SimulatedAnnealing(1e9, 0.99, x)),
        ("SA, 1e11 & 0.95", x => new SimulatedAnnealing(1e11, 0.95, x)),
        ("SA, 1e10 & 0.95", x => new SimulatedAnnealing(1e10, 0.95, x)),
        ("SA, 1e9 & 0.95", x => new SimulatedAnnealing(1e9, 0.95, x))
      ) : List[(String, NeuralNetworkOptimizationProblem => OptimizationAlgorithm)];

      val split = 0.75
      val iters = if (full) 50000 else 500
      val runs = 8
      val nodeList = List(10, 20, 30)
      runNeuralNetTestMatrix("faults", "faults-nn20.json", split, nodeList,
        runs, iters, faults, algos)
    }

    // This is to get a longer history for GA, and also to get timing
    // information.  It took around 9 hours to run on my machine.
    {
      val algos = List(
        //("RHC", x => new RandomizedHillClimbing(x))
        //("GA, 200, 100, 20", x => new StandardGeneticAlgorithm(200, 100, 20, x)),
        //("GA, 200, 100, 60", x => new StandardGeneticAlgorithm(200, 100, 60, x)),
        //("GA, 200, 140, 20", x => new StandardGeneticAlgorithm(200, 140, 20, x)),
        ("GA, 200, 140, 60", x => new StandardGeneticAlgorithm(200, 140, 60, x))
        //("GA, 200, 180, 20", x => new StandardGeneticAlgorithm(200, 180, 20, x)),
        //("GA, 200, 180, 60", x => new StandardGeneticAlgorithm(200, 180, 60, x))
      ) : List[(String, NeuralNetworkOptimizationProblem => OptimizationAlgorithm)];

      val split = 0.75
      val iters = if (full) 50000 else 500
      val runs = 1
      val nodeList = List(20)
      runNeuralNetTestMatrix("faults", "faults-ga-200-140-60.json", split, nodeList,
        runs, iters, faults, algos)
    }

    {
      val algos = List(
        ("RHC", x => new RandomizedHillClimbing(x)),
        ("GA, 200, 100, 20", x => new StandardGeneticAlgorithm(200, 100, 20, x)),
        ("GA, 200, 100, 60", x => new StandardGeneticAlgorithm(200, 100, 60, x)),
        ("GA, 200, 140, 20", x => new StandardGeneticAlgorithm(200, 140, 20, x)),
        ("GA, 200, 140, 60", x => new StandardGeneticAlgorithm(200, 140, 60, x)),
        ("GA, 200, 180, 20", x => new StandardGeneticAlgorithm(200, 180, 20, x)),
        ("GA, 200, 180, 60", x => new StandardGeneticAlgorithm(200, 180, 60, x))
      ) : List[(String, NeuralNetworkOptimizationProblem => OptimizationAlgorithm)];

      val split = 0.75
      val iters = if (full) 50000 else 500
      val runs = 1
      val nodeList = List(20)
      runNeuralNetTestMatrix("faults", "faults-nn19.json", split, nodeList,
        runs, iters, faults, algos)
    }
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
        (0 to 25).map(letterClass)) // output
    })
    val lettersCond = conditionAttribs(lettersRaw)
    val letters = if (full) lettersCond else
      lettersCond.take((0.05 * lettersCond.size).toInt)

    val lettersRows = letters.size
    println(f"Read $lettersRows%d rows.")

    val algos = List(
      ("RHC", x => new RandomizedHillClimbing(x))
    ) : List[(String, NeuralNetworkOptimizationProblem => OptimizationAlgorithm)];

    val split = 0.75
    val iters = if (full) 20000 else 500
    val runs = 1
    val nodeList = List(10, 20)
    runNeuralNetTestMatrix("letters", lettersOutput, split, nodeList,
      runs, iters, letters, algos)
  }

  // Modified version of KnapsackEvaluationFunction which counts calls
  class KnapsackEvalCount(w : Array[Double], v: Array[Double], maxV : Double,
    maxC : Array[Int]) extends KnapsackEvaluationFunction(w, v, maxV, maxC)
  {
    var calls = 0
    override def value(d : Instance) : Double =
    {
      calls += 1
      super.value(d)
    }
  }

  def knapsackTest()
  {
    // So far, this is a Scala translation of
    // ABAGAIL/src/opt/test/KnapsackTest.java
    val numItems       : Int    = 40
    val copiesEach     : Int    = 4
    val maxWeight      : Int    = 50
    val maxVolume      : Int    = 50
    val knapsackVolume : Double = maxVolume * numItems * copiesEach * 0.4

    val r = new scala.util.Random()
    val copies = Array.fill[Int](numItems)(copiesEach)
    val weights = (1 to numItems).map( _ => r.nextDouble * maxWeight).toArray
    val volumes = (1 to numItems).map( _ => r.nextDouble * maxVolume).toArray
    val ranges = Array.fill[Int](numItems)(copiesEach + 1)
    val ef = new KnapsackEvalCount(weights, volumes, knapsackVolume, copies)

    val odd = new DiscreteUniformDistribution(ranges)
    val nf = new DiscreteChangeOneNeighbor(ranges)
    val hcp = new GenericHillClimbingProblem(ef, odd, nf)

    val cf = new UniformCrossOver()
    val mf = new DiscreteChangeOneMutation(ranges)
    val gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

    // This tests all the algorithms with the 'optimal' settings:
    {
      val algos = List(
        ("RHC", 5000, () => new RandomizedHillClimbing(hcp)),
        ("SA, 1e2 & 0.999", 5000, () => new SimulatedAnnealing(1e2, 0.999, hcp)),
        ("GA, 200, 150, 25", 5000, () => new StandardGeneticAlgorithm(200, 150, 25, gap)),
        ("MIMIC, 200, 100", 5000, () => {
          val df = new DiscreteDependencyTree(.1, ranges)
          val pop = new GenericProbabilisticOptimizationProblem(ef, odd, df)
          new MIMIC(200, 100, pop)
        })
      ) : List[(String, Int, () => OptimizationAlgorithm)];
      // (Name, iters, algorithm)

      runOptimizationTestMatrix("knapsack", "knapsack-all.json", 20, ef, algos)
    }
    
    // This tests a variety of parameters and shows the tradeoff
    // between better results, and faster results.
    {
      val i = 30000
      val algos = List(
        ("SA, 1e3 & 0.9997", i, () => new SimulatedAnnealing(1e3, .9997, hcp)),
        ("SA, 1e4 & 0.9997", i, () => new SimulatedAnnealing(1e4, .9997, hcp)),
        ("SA, 1e4 & 0.9995", i, () => new SimulatedAnnealing(1e4, .9995, hcp)),
        ("SA, 1e2 & 0.999", i, () => new SimulatedAnnealing(1e2, .999, hcp)),
        ("SA, 1e3 & 0.999", i, () => new SimulatedAnnealing(1e3, .999, hcp)),
        ("SA, 1e4 & 0.999", i, () => new SimulatedAnnealing(1e4, .999, hcp)),
        ("SA, 1e5 & 0.999", i, () => new SimulatedAnnealing(1e5, .999, hcp)),
        ("SA, 1e6 & 0.999", i, () => new SimulatedAnnealing(1e6, .999, hcp)),
        ("SA, 1e7 & 0.999", i, () => new SimulatedAnnealing(1e7, .999, hcp)),
        ("SA, 1e8 & 0.999", i, () => new SimulatedAnnealing(1e8, .999, hcp)),
        ("SA, 1e4 & 0.995", i, () => new SimulatedAnnealing(1e5, .995, hcp)),
        ("SA, 1e4 & 0.99", i, () => new SimulatedAnnealing(1e5, .99, hcp))
      ) : List[(String, Int, () => OptimizationAlgorithm)];
      // (Name, iters, algorithm)

      runOptimizationTestMatrix("knapsack", "knapsack-sa.json", 40, ef, algos)
    }
  }

  // Modified version of TravelingSalesmanRouteEvaluationFunction
  // which counts calls
  class TspEvalCount(p : Array[Array[Double]]) extends
      TravelingSalesmanRouteEvaluationFunction(p)
  {
    var calls = 0
    override def value(d : Instance) : Double =
    {
      calls += 1
      super.value(d)
    }
  }

  def travelingSalesmanTest()
  {
    // Number of points:
    val n = 50
    val r = new scala.util.Random()
    val points = Array.tabulate[Double](n, 2) { (_,_) => r.nextDouble }

    val ef = new TspEvalCount(points)
    val dpd = new DiscretePermutationDistribution(n)
    val nf = new SwapNeighbor()
    val mf = new SwapMutation()
    val cf = new TravelingSalesmanCrossOver(ef)
    val hcp = new GenericHillClimbingProblem(ef, dpd, nf)
    val gap = new GenericGeneticAlgorithmProblem(ef, dpd, mf, cf)

    val ranges = Array.fill[Int](n)(n)

    // SA tuning
    {
      val i = 100000
      val algos = List(
        ("SA, 1e5 & 0.9997", i, () => new SimulatedAnnealing(1e5, .9997, hcp)),
        ("SA, 1e6 & 0.9997", i, () => new SimulatedAnnealing(1e6, .9997, hcp)),
        ("SA, 1e4 & 0.9995", i, () => new SimulatedAnnealing(1e4, .9995, hcp)),
        ("SA, 1e2 & 0.999", i, () => new SimulatedAnnealing(1e2, .999, hcp)),
        ("SA, 1e3 & 0.999", i, () => new SimulatedAnnealing(1e3, .999, hcp)),
        ("SA, 1e4 & 0.999", i, () => new SimulatedAnnealing(1e4, .999, hcp)),
        ("SA, 1e5 & 0.999", i, () => new SimulatedAnnealing(1e5, .999, hcp)),
        ("SA, 1e6 & 0.999", i, () => new SimulatedAnnealing(1e6, .999, hcp)),
        ("SA, 1e7 & 0.999", i, () => new SimulatedAnnealing(1e7, .999, hcp)),
        ("SA, 1e8 & 0.999", i, () => new SimulatedAnnealing(1e8, .999, hcp)),
        ("SA, 1e4 & 0.995", i, () => new SimulatedAnnealing(1e5, .995, hcp)),
        ("SA, 1e4 & 0.99", i, () => new SimulatedAnnealing(1e5, .99, hcp))
      ) : List[(String, Int, () => OptimizationAlgorithm)];
      // (Name, iters, algorithm)

      runOptimizationTestMatrix("tsp", "tspSa.json", 10, ef, algos)
    }

    {
      val algos = List(
        ("RHC", 20000, () => new RandomizedHillClimbing(hcp)),
        ("SA, 1e2 & 0.999", 20000, () => new SimulatedAnnealing(1e2, 0.999, hcp)),
        ("GA, 200, 150, 25", 20000, () => new StandardGeneticAlgorithm(200, 150, 25, gap)),
        ("MIMIC, 800, 200", 20000, () => {
          val df = new DiscreteDependencyTree(0.1, ranges)
          val odd = new DiscreteUniformDistribution(ranges)
          val pop = new GenericProbabilisticOptimizationProblem(ef, odd, df)
          new MIMIC(800, 200, pop)
        })
      ) : List[(String, Int, () => OptimizationAlgorithm)]

      runOptimizationTestMatrix("tsp", "tsp01.json", 8, ef, algos)
    }

    // MIMIC tuning
    {
      val algos = List(
        ("RHC", 5000, () => new RandomizedHillClimbing(hcp)),
        ("MIMIC, 800, 200", 5000, () => {
          val df = new DiscreteDependencyTree(0.1, ranges)
          val odd = new DiscreteUniformDistribution(ranges)
          val pop = new GenericProbabilisticOptimizationProblem(ef, odd, df)
          new MIMIC(800, 200, pop)
        }),
        ("MIMIC, 800, 400", 5000, () => {
          val df = new DiscreteDependencyTree(0.1, ranges)
          val odd = new DiscreteUniformDistribution(ranges)
          val pop = new GenericProbabilisticOptimizationProblem(ef, odd, df)
          new MIMIC(800, 400, pop)
        }),
        ("MIMIC, 400, 200", 5000, () => {
          val df = new DiscreteDependencyTree(0.1, ranges)
          val odd = new DiscreteUniformDistribution(ranges)
          val pop = new GenericProbabilisticOptimizationProblem(ef, odd, df)
          new MIMIC(400, 200, pop)
        })
      ) : List[(String, Int, () => OptimizationAlgorithm)]

      runOptimizationTestMatrix("tsp", "tspMimic.json", 4, ef, algos)
    }
  }

  // Modified version of ContinuousPeaksEvaluationFunction which
  // counts calls
  class CpEvalCount(t : Int) extends ContinuousPeaksEvaluationFunction(t)
  {
    var calls = 0
    override def value(d : Instance) : Double =
    {
      calls += 1
      super.value(d)
    }
  }
  
  def continuousPeaksTest()
  {
    val n : Int = 60
    val t : Int = n / 10
    val ranges = Array.fill[Int](n)(2)

    val ef = new CpEvalCount(t);
    val nf = new DiscreteChangeOneNeighbor(ranges);
    val mf = new DiscreteChangeOneMutation(ranges);
    val cf = new SingleCrossOver();
    val odd = new DiscreteUniformDistribution(ranges)
    val hcp = new GenericHillClimbingProblem(ef, odd, nf);
    val gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

    {
      val algos = List(
        ("RHC", 6000, () => new RandomizedHillClimbing(hcp)),
        ("SA, 1e9 & 0.97", 6000, () => new SimulatedAnnealing(1e9, 0.97, hcp)),
        ("GA, 200, 100, 10", 6000, () => new StandardGeneticAlgorithm(200, 100, 10, gap)),
        ("MIMIC, 200, 20", 6000, () => {
          val df = new DiscreteDependencyTree(0.1, ranges)
          val pop = new GenericProbabilisticOptimizationProblem(ef, odd, df)
          new MIMIC(200, 20, pop)
        })
      ) : List[(String, Int, () => OptimizationAlgorithm)]
      runOptimizationTestMatrix("continuous_peaks", "cp01.json", 10, ef, algos)
    }

  }

  // Run an entire matrix of neural network training tests.
  // name: Name of this overall test matrix (will appear in JSON)
  // filename: Output JSON file
  // split: The training/test split ratio
  // hiddenNodeList: Numbers of hidden nodes to try (one hidden layer only)
  // numRuns: Total number of runs to do
  // iters: Number of iterations to run
  // data: Dataset to use
  // algos: List of names & algorithms (by way of OptimizationAlgorithm)
  def runNeuralNetTestMatrix(
    name: String,
    filename: String,
    split: Double,
    hiddenNodeList: List[Int],
    numRuns: Int,
    iters: Int,
    data: List[Instance],
    algos: List[(String, NeuralNetworkOptimizationProblem => OptimizationAlgorithm)]
  )
  {
    // Get size of input & output:
    val inputs = data(0).size()
    val outputs = data(0).getLabel().size()
    println(s"Using $inputs inputs, $outputs outputs")
    val (train, test) = splitTrainTest(split, data)
    val trainSize = train.size
    val testSize = test.size
    println(s"Training: $trainSize, testing: $testSize")
    val results = algos.flatMap { case (algoName,algo) =>
      (1 to numRuns).flatMap { run =>
        hiddenNodeList.map { hiddenNodes =>
          val f: Future[List[ErrorResult]] = Future {
            val t0 = System.nanoTime()

            println(s"Starting $name, $algoName, run $run, $hiddenNodes nodes")
            val nodeCfg = Array(inputs, hiddenNodes, outputs)
            val factory = new BackPropagationNetworkFactory()
            val testNet = factory.createClassificationNetwork(nodeCfg, new LogisticSigmoid())
            val nets = optimizeNN(train, nodeCfg, iters, algo)
            val r = nets.map { case (iter,weights) =>
              val ctxt = TestParams(name, algoName, split, hiddenNodes, run, iter)
              testNet.setWeights(weights)
              val trainErr = nnBinaryError(train, testNet)
              val testErr = nnBinaryError(test, testNet)
              ErrorResult(ctxt, trainErr, testErr)
            }
            val t1 = (System.nanoTime() - t0) / 1e9
            println(f"Finished $name, $algoName, run $run ($t1%.2f seconds), $hiddenNodes nodes")
            r
          }
          f
        }
      }
    }

    val algoList = algos.map(_._1)
    val date = Calendar.getInstance().getTime()
    val testId = f"$name, started on $date, algorithms: $algoList, $split%.3f split, $iters iterations, hidden nodes tested: $hiddenNodeList"
    val writer = new PrintWriter(new File(filename))
    writeJsonHeader(writer, testId)

    var done = 0
    var failed = 0
    // This Future returns when all either have written or failed:
    val allWritten = Future.sequence (results.map { fut =>
      val written = fut.map { records =>
        writeJsonRecords(writer, records)
        val numRecs = records.size
        val numResults = results.size
        done = done + 1
        println(s"Wrote $numRecs records")
        println(s"$done done, $failed failed, of $numResults.")
      }
      written onFailure { case t =>
        done = done + 1
        failed = failed + 1
        val numResults = results.size
        println("Error with result: " + t.getMessage)
        println(s"$done done, $failed failed, of $numResults.")
      }
      written
    })

    // Only once all others are done (including writing), close the file:
    Await.result(allWritten, Duration.Inf)
    writeJsonEnd(writer)
    writer.close()
  }

  // Run an entire matrix of optimization tests.
  // name: Name of this overall test matrix (will appear in JSON)
  // filename: Output JSON file
  // numRuns: Total number of runs to do
  // ef: EvaluationFunction to run
  // algos: List of names, iterations, & algorithms
  def runOptimizationTestMatrix(
    name: String,
    filename: String,
    numRuns: Int,
    ef: EvaluationFunction,
    algos: List[(String, Int, () => OptimizationAlgorithm)]
  )
  {
    val results = algos.flatMap { case (algoName, iters, algoFn) =>
      (1 to numRuns).map { run =>
        val algo = algoFn()
        val f: Future[Iterable[OptimizationResult]] = Future {
          println(s"Starting $name, $algoName, run $run, $iters iterations")

          val t0 = System.nanoTime()
          val history = (1 to iters).flatMap { iter =>

            // Train, but isolate just the calls required for training:
            // (Yes, this is a kludge.)
            ef match {
              case ef2: KnapsackEvalCount => ef2.calls = 0
              case ef2: TspEvalCount => ef2.calls = 0
              case ef2: CpEvalCount => ef2.calls = 0
              case _ => ()
            }

            algo.train()

            val calls = ef match {
              case ef2: KnapsackEvalCount => ef2.calls
              case ef2: TspEvalCount => ef2.calls
              case ef2: CpEvalCount => ef2.calls
              case _ => 0
            }
            
            if (iter % 250 == 0) {
              val opt = ef.value(algo.getOptimal())
              println(f"$name, $algoName, run $run/$numRuns, $iter/$iters, $calls calls: $opt")
            }
            if (iter % 10 == 0) {
              val opt = ef.value(algo.getOptimal())
              Some(OptimizationResult(name, algoName, run, iter, calls, opt))
            } else None
          }
          val opt = ef.value(algo.getOptimal())
          val t1 = (System.nanoTime() - t0) / 1e9

          println(f"$name, $algoName, run $run, final ($t1%.2f seconds): $opt")
          history
        }
        f
      }
    }

    // TODO: The below is identical to runNeuralNetTestMatrix except
    // for some different status strings, and which JSON writing call
    // is used.  If I can get a single writeJsonRecords call with the
    // polymorphism I need (see comments near it), I can factor this
    // out easily.

    val algoList = algos.map(_._1)
    val date = Calendar.getInstance().getTime()
    val testId = f"$name, started on $date, runs: $numRuns, algorithms: $algoList"
    val writer = new PrintWriter(new File(filename))
    writeJsonHeader(writer, testId)

    var done = 0
    var failed = 0
    // This Future returns when all either have written or failed:
    val allWritten = Future.sequence (results.map { fut =>

      val written = fut.map { records =>
        writeJsonOptimizationResult(writer, records)
        val numRecs = records.size
        val numResults = results.size
        done = done + 1
        println(s"Wrote $numRecs records")
        println(s"$done done, $failed failed, of $numResults.")
      }
      written onFailure { case t =>
        done = done + 1
        failed = failed + 1
        val numResults = results.size
        println("Error with result: " + t.getMessage)
        println(s"$done done, $failed failed, of $numResults.")
      }
      written
    })

    // Only once all others are done (including writing), close the file:
    Await.result(allWritten, Duration.Inf)
    writeJsonEnd(writer)
    writer.close()
  }

  // Override implicit object CSVReader.open uses & change delimiter:
  def tabReader(fname : String) : CSVReader =
  {
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
  def nnBinaryError(set: DataSet, nn : BackPropagationNetwork) : Double =
  {
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
  // neural network, max number of iterations, and a function which
  // produces an OptimizationAlgorithm, construct a neural network and
  // train it one step at a time with the returned
  // OptimizationAlgorithm.  This returns a List of (iteration,
  // weights).
  def optimizeNN(
    set: DataSet,
    nodes: Iterable[Int],
    iters: Int,
    optFn: NeuralNetworkOptimizationProblem => OptimizationAlgorithm) :
      List[(Int, Array[Double])] =
  {
    // Build neural network & OptimizationAlgorithm:
    val (net, prob) = getNeuralNet(set, nodes)
    val opt = optFn(prob)

    (1 to iters).flatMap { i =>
      // Train the network one more step & use the updated weights:
      opt.train()
      val w = opt.getOptimal().getData()
      net.setWeights(w)
      if (i % 100 == 0) println(s"$i/$iters...")
      if (i % 10 == 0) Some((i, net.getWeights)) else None
    }.toList
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
      // TODO: Factor out all of the above.
      ("trainErr"    := jNumber(p.trainErr))       ->:
      ("testErr"     := jNumber(p.testErr))        ->:
      jEmptyObject
    })

  // Class giving the optimization value in some context
  case class OptimizationResult(test: String, name: String, run: Int,
      iter: Int, calls: Int, value: Double)

  implicit def OptimizationResultJson: EncodeJson[OptimizationResult] =
    EncodeJson((p: OptimizationResult) => {
      ("test"        := jString(p.test))  ->:
      ("name"        := jString(p.name))  ->:
      ("run"         := jNumber(p.run))   ->:
      ("calls"       := jNumber(p.calls)) ->:
      ("iter"        := jNumber(p.iter))  ->:
      // TODO: Factor out all of the above.
      ("value"       := jNumber(p.value)) ->:
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

  // Write the start of a JSON file into 'f', given a string for test
  // ID.  The next call must be 'writeJsonEnd' or 'writeJsonRecords'.
  // The point of this is to make a JSON file that jsonlite for R will
  // read.
  def writeJsonHeader(f: Writer, testId: String)
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
    f.write(",\"data\": [{}")
    // The commas must be right in the other records, but when I write
    // things in parallel I don't have any good way of specifying what
    // the "first" record is - so I can't just always write a leading
    // comma, or always write a trailing comma.  So, instead, I just
    // insert an initial dummy record, and then it doesn't matter.
  }

  // Write a list of ErrorResult to the JSON file; this assumes that
  // 'writeJsonHeader' has been called already, and assumes that you
  // will call 'writeJsonEnd' after all records are written.
  def writeJsonRecords(f: Writer, errors: Iterable[ErrorResult])
  {
    for (r <- errors) {
      f.write("," + r.asJson.spaces2)
      f.flush()
    }
  }

  // writeJsonRecords, but for OptimizationResult.  Yes, this is
  // atrocious, but I don't know Scala very well.  I need to figure
  // out how to make an Iterable that is full of items with some
  // trait.
  def writeJsonOptimizationResult(f: Writer, errors: Iterable[OptimizationResult])
  {
    for (r <- errors) {
      f.write("," + r.asJson.spaces2)
      f.flush()
    }
  }

  // Write the ending of the JSON data (after a call to
  // 'writeJsonHeader' and any number of calls to 'writeJsonRecords'),
  // and close the file.
  def writeJsonEnd(f: Writer)
  {
    f.write("]}")
    f.close()
  }

  // Normalize the given data to have mean of 0 and variance of 1.
  def normalize(data: Iterable[Double]) : Iterable[Double] =
  {
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
      List[Instance] =
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
    inNormed.zip(out).map { case (in, out) => instance(in, out) }.toList
  }

}
