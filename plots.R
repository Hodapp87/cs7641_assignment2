#!/usr/bin/env Rscript

###########################################################################
## Chris Hodapp (chodapp3)
## Georgia Institute of Technology
## CS7641, Machine Learning, Spring 2016
## Assignment 2, Randomized Optimization (2016-03-13)
###########################################################################

library(jsonlite);
library(ggplot2);

faults <- fromJSON("faults-nn-200runs.json");

faultsAgg <- aggregate(
    . ~ hiddenNodes + iter,
    subset(faults, select=c(testErr, trainErr, hiddenNodes, iter)),
    mean);

## Turn training error & testing error into separate entries:
faultsTrainErr <- data.frame(error       = faultsAgg$trainErr,
                             iter        = faultsAgg$iter,
                             hiddenNodes = faultsAgg$hiddenNodes,
                             stage       = "Train");
faultsTestErr <- data.frame(error       = faultsAgg$testErr,
                            iter        = faultsAgg$iter,
                            hiddenNodes = faultsAgg$hiddenNodes,
                            stage       = "Test");
faultsAgg2 = rbind(faultsTrainErr, faultsTestErr);
faultsAgg2["Hidden nodes"] <- sprintf("%d", faultsAgg2$hiddenNodes);

ggplot(data = faultsAgg2,
       aes(x=iter, y=error, group=interaction(stage, `Hidden nodes`))) +
    geom_line(aes(linetype=stage, colour=`Hidden nodes`)) +
    xlab("Iterations") +
    ylab("Error (ratio of incorrect classification)") +
    ggtitle("Learning curve (steel faults)");
