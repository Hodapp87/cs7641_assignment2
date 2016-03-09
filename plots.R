#!/usr/bin/env Rscript

###########################################################################
## Chris Hodapp (chodapp3)
## Georgia Institute of Technology
## CS7641, Machine Learning, Spring 2016
## Assignment 2, Randomized Optimization (2016-03-13)
###########################################################################

library(jsonlite);
library(ggplot2);

jsonDump <- fromJSON("faults-nn19.json");
data <- jsonDump$data
## data <- fromJSON("letters-nn-normed2.json");
## This decimates the plot, but I'm not sure if it does it well (it
## won't do it per-test):
## data <- data[data$name == "SA, 1e11 & 0.95" | data$name == "RHC",]
## data <- data[data$hiddenNodes == 20,]
skip <- 50
data10 <- data[seq(1,nrow(data),by=skip),]

dataAgg <- aggregate(
    . ~ hiddenNodes + iter + name,
    subset(data10, select=c(testErr, trainErr, hiddenNodes, iter, name)),
    mean);

dataStdev <- aggregate(
    . ~ hiddenNodes + iter + name,
    subset(data10, select=c(testErr, trainErr, hiddenNodes, iter, name)),
    sd);

## Turn training error & testing error into separate entries:
dataTrainErr <- data.frame(meanErr     = dataAgg$trainErr,
                           stdevErr    = dataStdev$trainErr,
                           iter        = dataAgg$iter,
                           hiddenNodes = dataAgg$hiddenNodes,
                           name        = dataAgg$name,
                           stage       = "Train");
dataTestErr <- data.frame(meanErr     = dataAgg$testErr,
                          stdevErr    = dataStdev$testErr,
                          iter        = dataAgg$iter,
                          hiddenNodes = dataAgg$hiddenNodes,
                          name        = dataAgg$name,
                          stage       = "Test");
dataAgg2 = rbind(dataTrainErr, dataTestErr);
dataAgg2["Hidden nodes"] <- sprintf("%d", dataAgg2$hiddenNodes);

ggplot(data = dataAgg2,
       aes(x=iter, y=meanErr, group=interaction(stage, name, hiddenNodes))) +
    geom_line(aes(linetype=stage, colour=interaction(name, hiddenNodes))) +
    geom_ribbon(aes(ymin=meanErr - stdevErr, ymax = meanErr + stdevErr, fill=interaction(name, hiddenNodes), alpha = 0.0)) +
    xlab("Iterations") +
    ylab("Error (ratio of incorrect classification)") +
    ggtitle(jsonDump$testId);



#     geom_ribbon(aes(ymin=

jsonDump <- fromJSON("knapsack01.json");
data <- jsonDump$data
data <- data[data$iter < 3000,]

dataAgg <- aggregate(. ~ iter + name,
                     subset(data, select=c(value, iter, name)),
                     mean);

dataStdev <- aggregate(. ~ iter + name,
                       subset(data, select=c(value, iter, name)),
                       sd);

## Turn training error & testing error into separate entries:
dataAgg2 <- data.frame(mean  = dataAgg$value,
                       stdev = dataStdev$value,
                       iter  = dataAgg$iter,
                       name  = dataAgg$name);

ggplot(data = dataAgg2,
       aes(x=iter, y=mean, group=interaction(name))) +
    geom_line(aes(colour=interaction(name))) +
    geom_ribbon(aes(ymin=mean - stdev, ymax = mean + stdev, fill=interaction(name), alpha = 0.1)) +
    xlab("Iterations") +
    ylab("Error (ratio of incorrect classification)") +
    ggtitle(jsonDump$testId);
