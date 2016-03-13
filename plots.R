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

## This is a wrapper around fromJSON that only loads from the JSON if
## an identically-named .Rda file doesn't exist, or if the 2nd
## argument is true.  If such an .Rda does exist, data is loaded from
## this instead; if not, 
fromJSONcached <- function(jsonFilename, reload = FALSE) {
    ## Attempt to remove the .json extension and add .Rda:
    rdaName <- paste(gsub(".json$", "", jsonFilename), ".Rda", sep="")
    if (file.exists(rdaName) & !reload) {
        return(local({
            load(rdaName);
            return(savedJsonData);
        }))
    } else {
        savedJsonData <- fromJSON(jsonFilename)
        save(savedJsonData, file = rdaName)
        return(savedJsonData)
    }
}
