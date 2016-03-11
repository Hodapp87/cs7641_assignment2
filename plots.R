#!/usr/bin/env Rscript

###########################################################################
## Chris Hodapp (chodapp3)
## Georgia Institute of Technology
## CS7641, Machine Learning, Spring 2016
## Assignment 2, Randomized Optimization (2016-03-13)
###########################################################################

library(jsonlite);
library(ggplot2);

getBinary <- function(val, digits) {
    if (digits > 0) {
        return(c(getBinary(val %/% 2, digits - 1), val %% 2))
    } else {
        return(c())
    }
}

contPeaks <- function(t, vec) {
    n0 <- 0
    count0 <- 0
    n1 <- 0
    count1 <- 0
    for (i in vec) {
        if (i == 0) {
            n1 <- max(count1, n1)
            count0 <- count0 + 1;
            count1 <- 0;
        } else if (i == 1) {
            n0 <- max(count0, n0)
            count0 <- 0;
            count1 <- count1 + 1;
        }
    }
    n0 <- max(count0, n0)
    n1 <- max(count1, n1)
    maxn <- max(n0, n1)
    tn <- if (n0 > t & n1 > t) { length(vec) } else 0
    fx <- maxn + tn
    return(data.frame(n0, n1, maxn, tn, fx))
}

contPeaksTable <- function(t, digits) {
    ## Generate 0...2^n:
    vals <- data.frame(x = c(0:(2^digits - 1)))
    ## Run 'contPeaks' on every column, producing a data frame from
    ## the individual ones it produces: (Open to suggestions for how to
    ## solve this more cleanly)
    fx <- do.call(rbind,
                  apply(vals, 1,
                        function(x) contPeaks(t, getBinary(x, digits))))
    fx$x <- vals
    return(fx)
}

jsonDump <- fromJSON("faults-nn20.json");
data <- jsonDump$data
## data <- fromJSON("letters-nn-normed2.json");
## This decimates the plot, but I'm not sure if it does it well (it
## won't do it per-test):
data <- data[data$name == "SA, 1e11 & 0.99" | data$name == "RHC",]
data <- data[data$hiddenNodes == 10,]
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
data <- data[data$iter < 1500,]

knapsack <- combineOptFrame(data)

ggplot(data = knapsack,
       aes(x=iter, y=mean, group=name)) +
    geom_line(aes(colour=name)) +
    geom_ribbon(aes(ymin=(mean - stdev), ymax = (mean + stdev), color=name, fill=name), linetype=2, alpha = 0.3) +
    xlab("Iterations") +
    ylab("Knapsack weight") +
    ggtitle(jsonDump$testId);

## Given a data frame 'data' read in from a JSON file produced by the
## Scala code for neural network training, perform some amount of
## decimation and aggregation, and split the training and testing
## error into separate rows.  The resultant frame will have columns
## 'meanErr' and 'stdevErr' for the mean and standard deviation of the
## error, rospectively, and a column 'stage' that is equal to either
## 'Train' or 'Test' depending on what type of error is in that row.
combineNnFrame <- function(data, skip = 1) {
    data <- data[seq(1,nrow(data),by=skip),];
    f <- . ~ hiddenNodes + iter + name;
    ss <- subset(data, select=c(testErr, trainErr, hiddenNodes, iter, name))
    means <- aggregate(f, ss, mean);
    stdevs <- aggregate(f, ss, sd);
    
    ## Turn training error & testing error into separate entries:
    train <- data.frame(meanErr     = means$trainErr,
                        stdevErr    = stdevs$trainErr,
                        iter        = means$iter,
                        hiddenNodes = means$hiddenNodes,
                        name        = means$name,
                        stage       = "Train");
    test  <- data.frame(meanErr     = means$testErr,
                        stdevErr    = stdevs$testErr,
                        iter        = means$iter,
                        hiddenNodes = means$hiddenNodes,
                        name        = means$name,
                        stage       = "Test");
    return(rbind(train, test));   
}

## This is basically combineNnFrame, but for the frames produced for
## optimization problems, which have no separate training & test
## (thus, the resultant from has no "stage" column).
combineOptFrame <- function(data, skip = 1) {
    data <- data[seq(1,nrow(data),by=skip),];
    f <- . ~ iter + name;
    ss <- subset(data, select=c(value, iter, name));
    means <- aggregate(f, ss, mean);
    stdevs <- aggregate(f, ss, sd);

    data2 <- data.frame(mean  = means$value,
                        stdev = stdevs$value,
                        iter  = means$iter,
                        name  = means$name);

    return(data2);
}

json16 <- fromJSON("faults-nn16.json");
json19 <- fromJSON("faults-nn19.json");
json20 <- fromJSON("faults-nn20.json");

data <- json20$data
data <- data[data$name == "RHC" | data$name == "SA, 1e10 & 0.95",]
data <- data[data$hiddenNodes == 20,]
faults20 <- combineNnFrame(data, 20);

data <- json16$data
## data <- data[data$name == "GA, 200, 140, 60",]
## data <- data[data$hiddenNodes == 20,]
faults16 <- combineNnFrame(data, 40);

ggplot(data = faults16,
       aes(x=iter, y=meanErr, group=interaction(stage, name, hiddenNodes))) +
    geom_line(aes(linetype=stage, colour=interaction(name, hiddenNodes))) +
    xlab("Iterations") +
    ylab("Error (ratio of incorrect classification)") +
    ggtitle(jsonDump$testId);

    geom_ribbon(aes(ymin=meanErr - stdevErr, ymax = meanErr + stdevErr, fill=interaction(name, hiddenNodes), colour=interaction(name, hiddenNodes)), linetype=2, alpha = 0.1) +
ggplot(data = faults20,



