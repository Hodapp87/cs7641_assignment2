Chris Hodapp (chodapp3), 2016-03-13
Georgia Institute of Technology, CS7641, Machine Learning, Spring 2016
Assignment 2: Randomized Optimization

This assignment was written in Scala and R (only for plotting and some
analysis).  It was run with R v3.2.3 on 64-bit Linux.

(TODO: What versions of Scala & SBT?)

The code (minus report & analysis code) is also available at:
https://github.com/Hodapp87/cs7641_assignment2

Short procedure for generating everything:
1. Install SBT (Scala Build Tool) and R.
2. Install packages in R with:
install.packages(c("ggplot2","jsonlite"));
3. Run "sbt run" and wait awhile.  A long, long while.
4. Run chodapp3-report.R to produce the plots and report.

Longer explanation:

The Scala code relies on the packages ABAGAIL
(https://github.com/pushkar/ABAGAIL), scala-csv
(https://github.com/tototoshi/scala-csv) and Argonaut
(http://argonaut.io/).  A compiled copy of ABAGAIL is included in the
'lib' directory of the source code, and the other libraries are
declared as dependencies in SBT (http://www.scala-sbt.org/).

To run the Scala code, install SBT and then run:
sbt run

This will pull all dependencies, compile the code and run it - which
in turn will load the included data files, condition it, perform all
of the actual training, produce some logging output, and several large
JSON files which contain information used for plots.  The .Rda files
are the same contents as the JSON files, but in R's native format.

The R code depends on these JSON files, as well as the CRAN packages
jsonlite and ggplot2.  The code itself is embedded in
'chodapp3-analysis.Rnw', a NoWeb file with LaTeX and R.

To make use of that (after running the Scala code), run
'chodapp3-report.R'.  That produces a final PDF - along with numerous
other intermediate files (LaTeX code, PDFs of graphs, and extracted R
code).

The data set is small enough that it is included alongside
the source code.  It is also available at
https://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults and this
is the source of the files 'Faults.NNA' and 'Faults27x7_var'.

If you want to get a proper estimate of the number of function
evaluations of different training methods, you will have to force SBT
to be single-threaded (due to the kludge that I used in order to count
evaluations).  Add the following after 'sbt' to do this:
-Dscala.concurrent.context.numThreads=1 -Dscala.concurrent.context.maxThreads=1

