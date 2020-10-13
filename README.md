# AppxSilhouette
Source code for PPS Sampling - Based Approximation Silhouette algorithm

This git includes both versions of the code developed for the work of Altieri, Pietracaprina, Pucci, Vandin: "Scalable Distributed Approximation of Internal Measures for Clustering Evaluation". (submitted to SDM21, arXiv https://arxiv.org/abs/2003.01430).

The "master" branch of this git features the code to be used as reference for the reviewers of SDM21 conference. Eventual other branches are intended to be experimental and not part of the submission.

## Description

This git contains both versions of the algorithm: Sequential and Map-Reduce.

## Sequential code

This code is developed by exploiting and customizing the Java-ml library from Abeel et al. (Abeel, T.; de Peer, Y. V. & Saeys, Y. Java-ML: A Machine Learning Library, Journal of Machine Learning Research, 2009, 10, 931-934), the original Java-ml source code is available at https://github.com/AbeelLab/javaml

## Usage
You can exploit the Main method contained in silhEval.research.ExperimentTests to perform tests and experiments

The Main admits various kinds of operations

### Random generation
example: java silhEval.research.ExperimentTests randomGen file outfile elements 100000 dimensions 3 outliers 4 minorRay 1 majorRay 10000 genseed 100

Generates a collection of comma separated double d-dimension values structured as a collection of n points inside a sphere of a certain radius (minorRay) and few outliers placed on the surface of a sphere of another radius (majorRay). Parameters:
  - file: relative path to output file
  - elements: number of elements of the inner sphere
  - minorRay: radius of the inner sphere
  - dimensions: dimensions of points
  - outliers: number of outliers
  - majorRay: radius of the outer sphere
  - genseed: seed for random generation
  
### Clusterize
example: java silhEval.research.ExperimentTests clusterize dataset infile output outfile clusters 10 limiter , ext kMedoids seed 100

Performs the clustering of a dataset using kMeans (if not specified) or kMedoids (if specified). Accepted file format is made by one element per line. Parameters:
  - dataset: relative path to input file
  - output: relative path to output file
  - clusters: number of clusters
  - limiter: character to be used as value separator. Default blank.
  - ext: if NOT present, an extra header line with some information will be written (only for reusing generated dataset with this Main)
  - kMedoids: if present, the clustering is performed using the provided implementation of the kMedoids algorithm. Elsewhere, kMeans is used
  - seed: seed for random generation
  
### Correctness test 1
example: java silhEval.research.ExperimentTests correctness confs 2 conf1 dataset1 conf2 datasets2 log repofile csv repo.csv epsilon 0.1 delta 0.1 t 1024 runs 100

Performs correctness tests of type 1: for each provided input, computes the exact silhouette, the PPS sampling based silhouette and the Uniform sampling based, observing errors in approximation and if approximated methods are capable to identify the best silhouette configurations. CSV accepted, last value is the cluester index. Parameters
  - confs: number of configurations to test
  - conf1, conf2, conf3...: relative paths to input files
  - log: relative path to output file
  - csv: relative path to output csv file
  - epsilon: PPS error threshold
  - delta: probability threshold
  - t: expected sample size (calculated using other parameters, if absent)
  - runs: number of trials

### Correctness test 2
example: java silhEval.research.ExperimentTests correctness2 confs 2 conf1 dataset1 conf2 datasets2 log repofile epsilon 0.1 delta 0.1 t 1024 runs 100

Performs correctness tests of type 1: for each provided input, computes the exact silhouette, the average of a number of PPS sampling based silhouette approximation rounds and the average of a number Uniform sampling based approximation rounds and the generalized simplified silhouette, observing errors in approximation and if approximated methods are capable to identify the best silhouette configurations. CSV accepted, last value is the cluester index. Parameters
  - confs: number of configurations to test
  - conf1, conf2, conf3...: relative paths to input files
  - log: relative path to output file
  - epsilon: PPS error threshold
  - delta: PPS error threshold
  - t: expected sample size (calculated using other parameters, if absent)
  - runs: number of approximation rounds for sampling-based methods

### Performance PPS
example: java silhEval.research.ExperimentTests performancePPS dataset inputfile log repofile csv repo.csv epsilon 0.1 delta 0.1 t 1024 append true

Performs a run of PPS sampling based silhouette approximation. Parameters:
  - dataset: relative path to input file
  - log: relative path to output file
  - csv: relative path to output csv file
  - epsilon: PPS error threshold
  - delta: PPS probability threshold
  - t: expected sample size (calculated using other parameters, if absent)
  - append: if true, appends the results to an existing csv file, elsewhere, a new file is created
  
### Performance Uniform
example: java silhEval.research.ExperimentTests performanceUniform dataset inputfile log repofile csv repo.csv epsilon 0.1 delta 0.1 t 1024 append true

Performs a run of Uniform sampling based silhouette approximation. Parameters:
  - dataset: relative path to input file
  - log: relative path to output file
  - csv: relative path to output csv file
  - epsilon: error threshold
  - delta: probability threshold
  - t: expected sample size (calculated using other parameters, if absent)
  - append: if true, appends the results to an existing csv file, elsewhere, a new file is created
  
### Exact only
example: java silhEval.research.ExperimentTests exactonly dataset inputfile log repofile csv repo.csv append true

Computes the exact silhouette using the baseline algorithm. Parameters:
  - dataset: relative path to input file
  - log: relative path to output file
  - csv: relative path to output csv file
  - append: if true, appends the results to an existing csv file, elsewhere, a new file is created
  

## Map-Reduce code
This code exploits the java Apache SPark official APIs and libraries.

## Usage
You can exploit the Main method contained in altierif.research.Main to perform tests and experiments

The Main admits various kinds of operations

### Generation
*** This functionality is actually deprecated, will be replaced by a new one, aligned with the sequential version ***

### importDb

example: spark-submit --class altierif.research.Main jarfile mode importDb  source infile path outfile

Imports a clustered dataset (format: csv with one header line, each line with elements values and cluster index) and stores it as a Dataset<Row> with features column saved as a DenseArray. Parameters:
  - source: relative path to the source file ON THE HDFS
  - path: relative path to the position ON HDFS where the Dataset object is stored
  
### importDb2

example: spark-submit --class altierif.research.Main jarfile mode importDb2  source infile path outfile

Imports a clustered dataset (format: csv with one header line, each line with elements values and cluster index) and stores it as a Dataset<Row> with features column saved as a simple array od doubles. Used to import datasets to be processed using Spark default API. Parameters:
  - source: relative path to the source file ON THE HDFS
  - path: relative path to the position ON HDFS where the Dataset object is stored

### Performance PPS
example: spark-submit --class altierif.research.Main jarfile mode performancePPS path datafile outfile csvout epsilon eps delta del t tvalue parsed true append false

Performs a run of PPS sampling based silhouette approximation. Parameters:
  - path: relative path to the position ON HDFS where the Dataset object is stored
  - outfile: relative path to output file (csv format)
  - epsilon: PPS error threshold
  - delta: PPS probability threshold
  - t: expected sample size (calculated using other parameters, if absent)
  - parsed: set it to true if the source contains a parsed Dataser<Row> object
  - append: if true, appends the results to an existing csv file, elsewhere, a new file is created
  
### Performance Spark
example: spark-submit --class altierif.research.Main jarfile mode performanceSpark performanceUniform path datafile outfile csvout

Performs a run of squared euclidean silhouette approximation, based on Spark's API Parameters. It only supports Dataset Imported using ImportDb2 mode:
  - path: relative path to the position ON HDFS where the Dataset object is stored
  - outfile: relative path to output file (csv format)
  - append: if true, appends the results to an existing csv file, elsewhere, a new file is created
  
### Performance Uniform
  
### Exact only

### Frahling-solher heuristic only
  
