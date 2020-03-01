package silhEval.research;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.Scanner;
import java.util.StringTokenizer;

import net.sf.javaml.clustering.Clusterer;
import net.sf.javaml.clustering.KMeans;
import net.sf.javaml.clustering.KMedoids;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;
import net.sf.javaml.distance.EuclideanDistance;
import net.sf.javaml.tools.data.FileHandler;
import silhEval.samplers.PPSSampler;
import silhEval.samplers.UniformSampler;
import silhEval.silhouettes.ExactSilhouette;
import silhEval.silhouettes.SamplingSilhouetteApprox;
import silhEval.silhouettes.SimplifiedSilhouette;
import silhEval.utils.Utils;
import silhEval.utils.Logger;

public class ExperimentTests {

	public static void main(String[] args) throws IOException {

		if ((Utils.index(args, "randomGen")) != -1) {
			FullRandomGen.main(args);
			System.out.println("Done!");
			return;
		}

		if ((Utils.index(args, "correctness")) != -1) {
			correctnessTest(args);
			System.out.println("Done!");
			return;
		}

		if ((Utils.index(args, "correctness2")) != -1) {
			correctnessTest2(args);
			System.out.println("Done!");
			return;
		}

		if ((Utils.index(args, "performance")) != -1) {
			performanceTest(args);
			System.out.println("Done!");
			return;
		}

		if ((Utils.index(args, "naive")) != -1) {
			performanceTestNaive(args);
			System.out.println("Done!");
			return;
		}

		if ((Utils.index(args, "exactOnly")) != -1) {
			exactSilhOnly(args);
			System.out.println("Done!");
			return;
		}

		if ((Utils.index(args, "clusterize")) != -1) {
			clusterize(args);
			System.out.println("Done!");
			return;
		}
	}

	private static void clusterize(String[] args) throws IOException {
		/*
		 * arguments parsing
		 */

		int index;
		String dataFile;
		String outFile;
		int k;
		int dimLimit;
		boolean limited = false;
		long clustSeed;
		Clusterer clusterer;
		String delemiter;
		boolean ext = false;

		if ((index = Utils.index(args, "dataset")) != -1) {
			dataFile = args[index + 1];
		} else {
			throw new IllegalArgumentException("no dataset source");
		}

		if ((index = Utils.index(args, "clusters")) != -1) {
			k = Integer.parseInt(args[index + 1]);
		} else {
			throw new IllegalArgumentException("no clusters");
		}

		if ((index = Utils.index(args, "ext")) != -1) {
			ext = true;
		}

		if ((index = Utils.index(args, "dimLimit")) != -1) {
			dimLimit = Integer.parseInt(args[index + 1]);
			limited = true;
		} else {
			dimLimit = Integer.MAX_VALUE;
		}

		if ((index = Utils.index(args, "limiter")) != -1) {
			delemiter = (args[index + 1]);
		} else {
			delemiter = " ";
		}

		if ((index = Utils.index(args, "output")) != -1) {
			outFile = args[index + 1];
		} else {
			outFile = dataFile.substring(0, dataFile.length() - 4) + "clK" + k + ".csv";
		}

		if ((index = Utils.index(args, "seed")) != -1) {
			clustSeed = Long.parseLong(args[index + 1]);
		} else {
			clustSeed = System.currentTimeMillis();
		}

		if ((index = Utils.index(args, "kMedoids")) != -1) {
			clusterer = new KMedoids(k, 100, new EuclideanDistance());
			//			clusterer = new FarthestFirst(k, new EuclideanDistance());
		} else {
			clusterer = new KMeans(k);
			((KMeans) clusterer).setSeed(clustSeed);
		}		

		Dataset data;
		data = FileHandler.loadDataset(new File(dataFile), -1, delemiter);

		if (limited) {
			for (int j = 0; j < data.size(); j++) {
				Iterator<Double> iter = data.get(j).values().iterator();
				for (int i = 0; i < dimLimit; i++) {
					System.out.println(i);
					iter.next();
				}
				while (iter.hasNext()) {
					iter.remove();
				}
			}
		}

		int dimensions = data.get(0).values().size();

		System.out.println("Elements = " + data.size());
		System.out.println("Dimensions = " + dimensions);
		System.out.println("Clustering with k = " + k);		

		Dataset[] clusters = clusterer.cluster(data);

		System.out.println("Writing file");

		Logger out = new Logger(outFile, false);

		for (int i = 0; i < dimensions; i++) {
			out.write("element" + i + ",");
		}

		out.writeLn("cluster");

		if (!ext) {

			out.write(dimensions + "," + k);

			for (int i = 0; i < dimensions - 2; i++) {
				out.write(",fooline");
			}
			out.writeLn("");
		}



		for (int i = 0; i < clusters.length; i++) {
			for (Iterator<Instance> iterator = clusters[i].iterator(); iterator.hasNext();) {
				Instance instance = (Instance) iterator.next();
				for (Iterator<Double> iterator2 = instance.values().iterator(); iterator2.hasNext();) {
					Double value = (Double) iterator2.next();
					out.write(value.toString() + ",");
				}
				out.writeLn(Integer.toString(i));
			}
		}

		out.close();

		System.out.println("Done!");

	}

	private static void performanceTest(String[] args) throws IOException {

		int index = -1;

		String dataFile = null;
		String logFile = null;
		String csvFile = null;
		double eps;
		double delta;
		boolean append = false;

		/*
		 * arguments parsing
		 */

		if ((index = Utils.index(args, "dataset")) != -1) {
			dataFile = args[index + 1];
		} else {
			throw new IllegalArgumentException("no dataset source");
		}

		if ((index = Utils.index(args, "log")) != -1) {
			logFile = args[index + 1];
		} else {
			logFile = "Experiment_" + System.currentTimeMillis() + ".log";
		}

		if ((index = Utils.index(args, "csv")) != -1) {
			csvFile = args[index + 1];
		} else {
			csvFile = "Experiment_" + System.currentTimeMillis() + "CSV.csv";
		}

		if ((index = Utils.index(args, "epsilon")) != -1) {
			eps = Double.parseDouble(args[index + 1]);
		} else {
			eps = 0.01;
		}

		if ((index = Utils.index(args, "delta")) != -1) {
			delta = Double.parseDouble(args[index + 1]);
		} else {
			delta = 0.01;
		}

		if ((index = Utils.index(args, "append")) != -1) {
			append = true;
		}

		/*
		 * Dataset loading
		 */

		Logger log = new Logger(logFile, false);
		Logger csvStats = new Logger(csvFile, append);

		Dataset[] clusters = parseCSVData(dataFile);

		int n = 0;

		for (int i = 0; i < clusters.length; i++) {
			n = n + clusters[i].size();
		}

		int k = clusters.length;

		log.writeLn("***EXPERIMENT RESUME***");
		log.writeLn("\n");
		log.writeLn("***\n");
		log.writeLn("Total dataset size: " + n + "\n");
		log.writeLn("Absolute error threshold for single element silhouette: " + eps);
		log.writeLn("Probability to have a higher error: " + delta);
		log.write("Sampling probability strategy: PPS");

		log.writeLn("\n");
		log.writeLn("***\n");

		System.out.println("Total dataset size: " + n);

		if (!append) {

			csvStats.write("type,n,k,t,");
			for (int i = 0; i < clusters.length; i++) {
				csvStats.write("n" + (i + 1) + ",e" + (i + 1) + ",s" + (i + 1) + ",");
			}
			csvStats.writeLn("delta,eps,estimatedSilh,sampTime,calcTimes,overallTimes");
		}

		log.writeLn("***\n");

		int t = (int) Math
				.ceil((1 / (2 * Math.pow(eps, 2))) * (Math.log(4) + Math.log(n) + Math.log(k) - Math.log(delta)));

		System.out.println("Sampling phase - Minumum expected sample size: " + t);

		/*
		 * Step 1: sampling phase
		 */

		long startCpuTimeNanoSampling = Utils.getUserTime();

		Dataset[] samples = new Dataset[clusters.length];

		for (int i = 0; i < samples.length; i++) {
			samples[i] = new DefaultDataset();
		}

		double[][] samplingProbabs = new double[clusters.length][];

		double[][] allpr = new double[clusters.length][];

		samplingProbabs = PPSSampler.sample(clusters, samples, eps, delta, t, allpr);

		double samplingTime = ((Utils.getUserTime() - startCpuTimeNanoSampling) / 10e8);

		log.writeLn("Expected sample size: " + t + "\n");

		for (int i = 0; i < samples.length; i++) {
			log.writeLn("Sampling for cluster " + (i + 1) + ":");
			log.writeLn("Original cluster size: " + clusters[i].size());
			log.writeLn("cluster sample size: " + samples[i].size() + "\n");
		}

		/*
		 * Step 2: silhouette
		 */

		SamplingSilhouetteApprox appx = new SamplingSilhouetteApprox();

		long startCpuTimeNanoSilhEval = Utils.getUserTime();

		double silhouette = appx.score(clusters, samples, samplingProbabs);

		double calcTime = (Utils.getUserTime() - startCpuTimeNanoSilhEval) / 10e8;
		double overallTime = (Utils.getUserTime() - startCpuTimeNanoSampling) / 10e8;

		log.writeLn("silhouette calculated with PPS sampling-based algorithm: " + silhouette + "\n");

		log.writeLn("elapsed CPU time Sampling Phase: " + samplingTime + "s");
		log.writeLn("elapsed CPU time for silhouette calculation: " + calcTime + "s");
		log.writeLn("overall elapsed CPU time: " + overallTime + "s");

		System.out.println(" Sampling-based algorithm: " + silhouette);

		csvStats.write("PPS," + n + "," + k + "," + t + ",");
		for (int i = 0; i < clusters.length; i++) {

			double sumPr = 0.0;
			for (int j = 0; j < clusters[i].size(); j++) {
				sumPr = sumPr + allpr[i][j];
			}
			csvStats.write(clusters[i].size() + "," + sumPr + "," + samples[i].size() + ",");
		}
		csvStats.writeLn(
				delta + "," + eps + "," + silhouette + "," + samplingTime + "," + calcTime + "," + overallTime);

		log.writeLn("***");
		log.writeLn("RESUME");
		log.writeLn("***");
		log.writeLn("Times: ");
		log.writeLn("average sampling time: " + samplingTime);
		log.writeLn("average appx silhouette calculation time: " + calcTime);
		log.writeLn("average overall time: " + overallTime + "\n");

		log.writeLn("Measures: ");
		log.writeLn("average approximated silhouette: " + silhouette);

		log.close();

		csvStats.close();

		System.out.println("Done!");

	}

	private static void exactSilhOnly(String[] args) throws IOException {

		int index = -1;
		String dataFile;
		double exactSilhouette = 0;
		String logFile;
		String csvFile;
		boolean append = false;

		if ((index = Utils.index(args, "dataset")) != -1) {
			dataFile = args[index + 1];
		} else {
			throw new IllegalArgumentException("no dataset source");
		}

		if ((index = Utils.index(args, "log")) != -1) {
			logFile = (args[index + 1]);
		} else {
			logFile = ("Experiment_" + System.currentTimeMillis() + ".log");
		}

		if ((index = Utils.index(args, "csv")) != -1) {
			csvFile = args[index + 1];
		} else {
			csvFile = "Experiment_" + System.currentTimeMillis() + "CSV.csv";
		}

		if ((index = Utils.index(args, "append")) != -1) {
			append = true;
		}

		Logger log = new Logger(logFile, false);
		Logger csvStats = new Logger(csvFile, append);

		Dataset[] clusters = parseCSVData(dataFile);

		int n = 0;

		for (int i = 0; i < clusters.length; i++) {
			n = n + clusters[i].size();
		}

		int k = clusters.length;

		log.writeLn("Exact silhouette computation for test dataset " + dataFile + "\n");

		long startCpuTimeNano = Utils.getUserTime();
		ExactSilhouette exact = new ExactSilhouette();
		exactSilhouette = exact.score(clusters);

		double taskCPUTime = (Utils.getUserTime() - startCpuTimeNano) / 10e8;

		log.writeLn("silhouette calculated with exact algorithm: " + exactSilhouette + "\n");
		log.writeLn("elapsed CPU time for exact silhouette calculation: " + taskCPUTime + " s" + "\n");

		System.out.println("silhouette calculated with exact algorithm: " + exactSilhouette + "\n");

		if (!append) {
			csvStats.writeLn("type,N,k,silhouette,CPUTime");
		}
		csvStats.writeLn("exact," + n + "," + k + "," + exactSilhouette + "," + taskCPUTime);

		log.close();
		csvStats.close();

	}

	private static void correctnessTest(String[] args) throws IOException {

		int index = -1;

		String[] dataFiles = null;
		String logFile = null;
		String csvFile = null;
		double eps;
		double delta;
		int runs = 1;
		int t;

		/*
		 * arguments parsing
		 */

		if ((index = Utils.index(args, "confs")) != -1) {
			dataFiles = new String[Integer.parseInt(args[index + 1])];
		} else {
			throw new IllegalArgumentException("no number of confs");
		}

		for (int i = 0; i < dataFiles.length; i++) {
			if ((index = Utils.index(args, "conf"+(i+1))) != -1) {
				dataFiles[i] = args[index + 1];
			}else {
				throw new IllegalArgumentException("no conf"+(i+1));
			}
		}

		if ((index = Utils.index(args, "log")) != -1) {
			logFile = args[index + 1];
		} else {
			logFile = "Experiment_" + System.currentTimeMillis() + ".log";
		}

		if ((index = Utils.index(args, "csv")) != -1) {
			csvFile = args[index + 1];
		} else {
			csvFile = "Experiment_" + System.currentTimeMillis() + "CSV.csv";
		}

		if ((index = Utils.index(args, "epsilon")) != -1) {
			eps = Double.parseDouble(args[index + 1]);
		} else {
			eps = 0.01;
		}

		if ((index = Utils.index(args, "delta")) != -1) {
			delta = Double.parseDouble(args[index + 1]);
		} else {
			delta = 0.01;
		}

		if ((index = Utils.index(args, "runs")) != -1) {
			runs = Integer.parseInt(args[index + 1]);
		}

		if ((index = Utils.index(args, "t")) != -1) {
			t = Integer.parseInt(args[index + 1]);
		} else {
			t = 0;
		} 

		/*
		 * Dataset loading
		 */

		Logger log = new Logger(logFile, false);
		Logger csvStats = new Logger(csvFile, false);

		Dataset[][] confs = new Dataset[dataFiles.length][];

		for (int j = 0; j < confs.length; j++) {
			confs[j] = parseCSVData(dataFiles[j]);
		}

		double[] costs = new double[dataFiles.length];

		for (int i = 0; i < costs.length; i++) {
			costs[i] = KMeans.getOverallKMeansCostFunction(confs[i], new EuclideanDistance());
		}

		int n = 0;

		for (int i = 0; i < confs[0].length; i++) {
			n = n + confs[0][i].size();
		}

		int[] ks = new int[confs.length];

		for (int i = 0; i < ks.length; i++) {
			ks[i] = confs[i].length;
		}

		log.writeLn("***EXPERIMENT RESUME***");
		log.writeLn("\n");
		log.writeLn("***\n");
		log.writeLn("Total dataset size: " + n + "\n");
		log.writeLn("Absolute error threshold for single element silhouette: " + eps);
		log.writeLn("Probability to have a higher error: " + delta);
		log.writeLn("Approximated silhouette runs: " + runs);
		log.writeLn("\n");
		log.writeLn("***\n");

		System.out.println("Total dataset size: " + n);

		csvStats.write("type,n,delta,eps,");

		for (int i = 0; i < confs.length; i++) {
			csvStats.write("k"+(i+1)+",t"+(i+1)+",");		
		}

		for (int i = 0; i < confs.length; i++) {
			for (int j = 0; j < confs[i].length; j++) {
				csvStats.write("n" + (j + 1) + ",s" + (j + 1) + ",");
			}
			csvStats.write("estimatedSilh"+(i+1)+",error"+(i+1)+",***,");
		}
		csvStats.writeLn("choice");

		int[] ts = new int[confs.length];

		/*
		 * Exact computation
		 */
		double[] exactSilhouettes = new double[confs.length];

		for (int i = 0; i < exactSilhouettes.length; i++) {
			ExactSilhouette exact = new ExactSilhouette();
			exactSilhouettes[i] = exact.score(confs[i]);
			if (t == 0) {
				ts[i] = (int) Math.ceil((1 / (2 * Math.pow(eps, 2))) * (Math.log(4) + Math.log(n) + Math.log(ks[i]) - Math.log(delta)));
			} else {
				ts[i] = t;
			}
			log.writeLn("Exact silhouette computation for test dataset " + dataFiles[i]);
			log.writeLn("silhouette calculated with exact algorithm: " + exactSilhouettes[i]);		
			log.writeLn("kmeans cost function (sum): " + costs[i] +"\n");
		}

		log.writeLn("***\n");

		csvStats.write("exact," + n + ","+ delta + "," + eps + ",");

		for (int i = 0; i < confs.length; i++) {
			csvStats.write(ks[i]+","+ts[i]+",");
			for (int j = 0; j < confs[i].length; j++) {
				csvStats.write(confs[i][j].size()+",0,");
			}
			csvStats.write(exactSilhouettes[i] + ","+costs[i]+",***,");			
		}

		int realBest = 0;
		double max = -2.0;
		for (int i = 0; i < exactSilhouettes.length; i++) {
			if (exactSilhouettes[i] > max) {
				max = exactSilhouettes[i];
				realBest = i;
			}
		}
		csvStats.writeLn(""+(realBest+1));
		log.writeLn("best choice: " + (realBest+1));

		System.out.println("Sampling phase - Minumum expected sample sizes: ");
		log.writeLn("Minimum expected sample size: ");

		for (int i = 0; i < confs.length; i++) {
			System.out.println(ts[i]);
			log.writeLn(""+ts[i]);
		}

		System.out.println("best choice: " + (realBest+1));

		/**************** PPS *****************/
		log.writeLn("*** PPS SAMPLING ***\n");

		int[] choicesPPS = new int[runs];
		int[] choicesUnif = new int[runs];

		int correctPPS = 0;
		int correctUnif = 0;

		double appxSilhouettesPPS[][] = new double[runs][confs.length];
		double appxSilhouettesUnif[][] = new double[runs][confs.length];

		double errorsPPS[][] = new double[runs][confs.length];
		double errorsUnif[][] = new double[runs][confs.length];

		for (int i = 0; i < runs; i++) {
			log.writeLn("*** Round: " + (i + 1) + "***\n");
			appxSilhouettesPPS[i] = correctnessAppx(confs, eps, delta, exactSilhouettes, log, csvStats, n, ts, false);

			choicesPPS[i] = -1;
			max = -2.0;

			for (int j = 0; j < appxSilhouettesPPS[j].length; j++) {

				errorsPPS[i][j] = Math.abs(exactSilhouettes[j] - appxSilhouettesPPS[i][j]);

				if(appxSilhouettesPPS[i][j] > max) {
					max = appxSilhouettesPPS[i][j];
					choicesPPS[i] = j;
				}
			}
			log.writeLn("choice:"+(choicesPPS[i] + 1)+"\n***");
			csvStats.writeLn(""+(choicesPPS[i]+1));
			System.out.println("PPS round "+ (i + 1) +": "+ (choicesPPS[i]+1));

			if (choicesPPS[i] == realBest) {
				correctPPS++;
			}
		}

		/**************** Uniform *****************/
		log.writeLn("*** Uniform SAMPLING ***\n");

		for (int i = 0; i < runs; i++) {
			log.writeLn("*** Round: " + (i + 1) + "***\n");
			appxSilhouettesUnif[i] = correctnessAppx(confs, eps, delta, exactSilhouettes, log, csvStats, n, ts, true);
			choicesUnif[i] = -1;
			max = -2.0;

			for (int j = 0; j < appxSilhouettesUnif[j].length; j++) {

				errorsUnif[i][j] = Math.abs(exactSilhouettes[j] - appxSilhouettesUnif[i][j]);

				if(appxSilhouettesUnif[i][j] > max) {
					max = appxSilhouettesUnif[i][j];
					choicesUnif[i] = j;
				}
			}
			log.writeLn("choice:"+(choicesUnif[i] + 1)+"\n***");
			csvStats.writeLn(""+(choicesUnif[i]+1));
			System.out.println("Uniform round "+ (i + 1) +": "+ (choicesUnif[i]+1));

			if (choicesUnif[i] == realBest) {
				correctUnif++;
			}
		}

		double ratioPPS = ((double)correctPPS/(double)runs)*100;		

		log.writeLn("PPS made the correct choice " +correctPPS+ " times,  in percentage: "+ratioPPS+ "%");

		double ratioUnif = ((double)correctUnif/(double)runs)*100;

		log.writeLn("Uniform made the correct choice " +correctUnif+ "  in percentage: "+ratioUnif+ "% \n");

		double[] avgErrorsPPS = new double[confs.length];
		double[] maxErrorsPPS = new double[confs.length];
		double[] varErrorsPPS = new double[confs.length];

		for (int i = 0; i < avgErrorsPPS.length; i++) {
			avgErrorsPPS[i] = 0;
			maxErrorsPPS[i] = 0;
			for (int j = 0; j < runs; j++) {

				avgErrorsPPS[i] = avgErrorsPPS[i] + errorsPPS[j][i];

				if (errorsPPS[j][i] > maxErrorsPPS[i]) {
					maxErrorsPPS[i] = errorsPPS[j][i];
				}

			}
			avgErrorsPPS[i] = avgErrorsPPS[i]/runs;

			for (int j = 0; j < runs; j++) {
				varErrorsPPS[i] = varErrorsPPS[i] + Math.pow((errorsPPS[j][i] - avgErrorsPPS[i]), 2);
			}
			varErrorsPPS[i] = varErrorsPPS[i]/runs;
		}

		double[] avgErrorsUnif = new double[confs.length];
		double[] maxErrorsUnif = new double[confs.length];
		double[] varErrorsUnif = new double[confs.length];

		for (int i = 0; i < avgErrorsUnif.length; i++) {
			avgErrorsUnif[i] = 0;
			maxErrorsUnif[i] = 0;
			for (int j = 0; j < runs; j++) {

				avgErrorsUnif[i] = avgErrorsUnif[i] + errorsUnif[j][i];

				if (errorsUnif[j][i] > maxErrorsUnif[i]) {
					maxErrorsUnif[i] = errorsUnif[j][i];
				}

			}
			avgErrorsUnif[i] = avgErrorsUnif[i]/runs;

			for (int j = 0; j < runs; j++) {
				varErrorsUnif[i] = varErrorsUnif[i] + Math.pow((errorsUnif[j][i] - avgErrorsUnif[i]), 2);
			}
			varErrorsUnif[i] = varErrorsUnif[i]/runs;
		}

		for (int i = 0; i < confs.length; i++) {
			log.writeLn("conf "+ (i+1)+ ":\n"
					+ "   PPS:\n"
					+ "       max error: "+maxErrorsPPS[i]+"\n"
					+ "       avg error: "+avgErrorsPPS[i]+"\n"
					+ "       variance: "+varErrorsPPS[i]+"\n"
					+ "   Uniform:\n"
					+ "       max error: "+maxErrorsUnif[i]+"\n"
					+ "       avg error: "+avgErrorsUnif[i]+"\n"
					+ "       variance: "+varErrorsUnif[i]+"\n");
		}

		log.close();

		csvStats.close();

		System.out.println("Done!");
	}

	private static double[] correctnessAppx(Dataset[][] confs, double eps, double delta,
			double[] exactSilhouettes, Logger log, Logger csvStats, int n, int[] ts, boolean naiveSampling) {

		double[] appxSilhouettes = new double[confs.length];
		double[] errors = new double[confs.length];

		Dataset[][] samples = new Dataset[confs.length][];

		for (int i = 0; i < confs.length; i++) {

			samples[i] = new Dataset[confs[i].length];

			for (int j = 0; j < samples[i].length; j++) {
				samples[i][j] = new DefaultDataset();
			}

			double[][] samplingProbabs = new double[confs[i].length][];

			if (naiveSampling) {
				samplingProbabs = UniformSampler.sample(confs[i], samples[i], ts[i]);
			} else {
				double[][] allpr = new double[confs[i].length][];
				samplingProbabs = PPSSampler.sample(confs[i], samples[i], eps, delta, ts[i], allpr);
			}

			for (int j = 0; j < samples[i].length; j++) {
				log.writeLn("Sampling for cluster " + (j + 1) + ":");
				log.writeLn("Original cluster size: " + confs[i][j].size());
				log.writeLn("cluster sample size: " + samples[i][j].size() + "\n");
			}
			/*
			 * Config 1: silhouette appx
			 */
			SamplingSilhouetteApprox appx = new SamplingSilhouetteApprox();
			appxSilhouettes[i] = appx.score(confs[i], samples[i], samplingProbabs);
			errors[i] = Math.abs(appxSilhouettes[i] - exactSilhouettes[i]);

			if (naiveSampling) {
				log.writeLn("silhouette 1 calculated with naive sampling algorithm: " + appxSilhouettes[i]);
			} else {
				log.writeLn("silhouette 1 calculated with PPS sampling-based algorithm: " + appxSilhouettes[i]);
			}
			log.writeLn("Absolute error: " + errors[i] + "\n");
		}

		if (naiveSampling) {
			csvStats.write("Uniform,");
		} else {
			csvStats.write("PPS,");
		}

		csvStats.write(n + "," + delta + "," + eps + ",");
		for (int i = 0; i < confs.length; i++) {
			csvStats.write(confs[i].length + "," + ts[i] + ",");
			for (int j = 0; j < confs[i].length; j++) {
				csvStats.write(confs[i][j].size() + "," + samples[i][j].size() + ",");
			}
			csvStats.write( + appxSilhouettes[i] + "," + errors[i] + ",***,");
		}

		return appxSilhouettes;


	}

	private static void correctnessTest2(String[] args) throws IOException {

		int index = -1;

		String[] dataFiles = null;
		String logFile = null;
		double eps;
		double delta;
		int runs = 1;
		int t;

		/*
		 * arguments parsing
		 */

		if ((index = Utils.index(args, "confs")) != -1) {
			dataFiles = new String[Integer.parseInt(args[index + 1])];
		} else {
			throw new IllegalArgumentException("no number of confs");
		}

		for (int i = 0; i < dataFiles.length; i++) {
			if ((index = Utils.index(args, "conf"+(i+1))) != -1) {
				dataFiles[i] = args[index + 1];
			}else {
				throw new IllegalArgumentException("no conf"+(i+1));
			}
		}

		if ((index = Utils.index(args, "log")) != -1) {
			logFile = args[index + 1];
		} else {
			logFile = "Experiment_" + System.currentTimeMillis() + ".log";
		}

		if ((index = Utils.index(args, "epsilon")) != -1) {
			eps = Double.parseDouble(args[index + 1]);
		} else {
			eps = 0.01;
		}

		if ((index = Utils.index(args, "delta")) != -1) {
			delta = Double.parseDouble(args[index + 1]);
		} else {
			delta = 0.01;
		}

		if ((index = Utils.index(args, "runs")) != -1) {
			runs = Integer.parseInt(args[index + 1]);
		}

		if ((index = Utils.index(args, "t")) != -1) {
			t = Integer.parseInt(args[index + 1]);
		} else {
			t = 0;
		} 

		/*
		 * Dataset loading
		 */

		Logger log = new Logger(logFile, false);

		Dataset[][] confs = new Dataset[dataFiles.length][];

		for (int j = 0; j < confs.length; j++) {
			confs[j] = parseCSVData(dataFiles[j]);
		}

		double[] costs = new double[dataFiles.length];

		for (int i = 0; i < costs.length; i++) {
			costs[i] = KMeans.getOverallKMeansCostFunction(confs[i], new EuclideanDistance());
		}

		int n = 0;

		for (int i = 0; i < confs[0].length; i++) {
			n = n + confs[0][i].size();
		}

		int[] ks = new int[confs.length];

		for (int i = 0; i < ks.length; i++) {
			ks[i] = confs[i].length;
		}

		log.writeLn("***EXPERIMENT RESUME***");
		log.writeLn("\n");
		log.writeLn("***\n");
		log.writeLn("Total dataset size: " + n + "\n");
		log.writeLn("Absolute error threshold for single element silhouette: " + eps);
		log.writeLn("Probability to have a higher error: " + delta);
		log.writeLn("Approximated silhouette runs: " + runs);
		log.writeLn("\n");
		log.writeLn("***\n");

		System.out.println("Total dataset size: " + n);

		int[] ts = new int[confs.length];

		/*
		 * Exact computation
		 */
		double[] exactSilhouettes = new double[confs.length];

		for (int i = 0; i < exactSilhouettes.length; i++) {
			ExactSilhouette exact = new ExactSilhouette();
			exactSilhouettes[i] = exact.score(confs[i]);
			if (t == 0) {
				ts[i] = (int) Math.ceil((1 / (2 * Math.pow(eps, 2))) * (Math.log(4) + Math.log(n) + Math.log(ks[i]) - Math.log(delta)));
			} else {
				ts[i] = t;
			}
			log.writeLn("Exact silhouette computation for test dataset " + dataFiles[i]);
			log.writeLn("silhouette calculated with exact algorithm: " + exactSilhouettes[i]);		
			log.writeLn("kmeans cost function (sum): " + costs[i] +"\n");
		}

		log.writeLn("***\n");

		int realBest = 0;
		double max = -2.0;
		for (int i = 0; i < exactSilhouettes.length; i++) {
			if (exactSilhouettes[i] > max) {
				max = exactSilhouettes[i];
				realBest = i;
			}
		}
		log.writeLn("best choice: " + (realBest+1));

		System.out.println("Sampling phase - Minumum expected sample sizes: ");
		log.writeLn("Minimum expected sample size: ");

		for (int i = 0; i < confs.length; i++) {
			System.out.println(ts[i]);
			log.writeLn(""+ts[i]);
		}

		System.out.println("best choice: " + (realBest+1));

		/********SIMPLIFIED SILHOUETTE*********/

		double[] simplifiedSilhouettes = new double[confs.length];

		for (int i = 0; i < simplifiedSilhouettes.length; i++) {
			SimplifiedSilhouette simpl = new SimplifiedSilhouette();
			simplifiedSilhouettes[i] = simpl.score(confs[i]);
			log.writeLn("Simplified silhouette computation for test dataset " + dataFiles[i]);
			log.writeLn("silhouette calculated with simplified algorithm: " + simplifiedSilhouettes[i]);
			log.writeLn("kmeans cost function (sum): " + costs[i] +"\n");
		}

		log.writeLn("***\n");

		int simplBest = 0;
		max = -2.0;
		for (int i = 0; i < simplifiedSilhouettes.length; i++) {
			if (simplifiedSilhouettes[i] > max) {
				max = simplifiedSilhouettes[i];
				simplBest = i;
			}
		}
		log.writeLn("Choice: " + (simplBest+1));



		System.out.println("Sampling phase - Minumum expected sample sizes: ");
		log.writeLn("Minimum expected sample size: ");

		for (int i = 0; i < confs.length; i++) {
			System.out.println(ts[i]);
			log.writeLn(""+ts[i]);
		}

		System.out.println("best choice: " + (simplBest+1));

		int choicePPS;
		int choiceUnif;

		double appxSilhouettesPPS[] = new double[confs.length];
		double appxSilhouettesUnif[] = new double[confs.length];

		double errorsPPS[] = new double[confs.length];
		double errorsUnif[] = new double[confs.length];

		/**************** PPS *****************/
		log.writeLn("*** PPS SAMPLING ***\n");

		for (int i = 0; i < confs.length; i++) {
			appxSilhouettesPPS[i] = correctnessAppx2(confs[i], eps, delta, runs, exactSilhouettes[i], log, n, ts[i], false);
		}

		choicePPS = -1;
		max = -2.0;

		for (int j = 0; j < appxSilhouettesPPS.length; j++) {

			errorsPPS[j] = Math.abs(exactSilhouettes[j] - appxSilhouettesPPS[j]);

			if(appxSilhouettesPPS[j] > max) {
				max = appxSilhouettesPPS[j];
				choicePPS = j;
			}
		}

		log.writeLn("choice PPS:"+(choicePPS + 1)+"\n***");
		System.out.println("PPS: "+ (choicePPS+1));

		/**************** Uniform *****************/
		log.writeLn("*** Uniform SAMPLING ***\n");

		for (int i = 0; i < confs.length; i++) {
			appxSilhouettesUnif[i] = correctnessAppx2(confs[i], eps, delta, runs, exactSilhouettes[i], log, n, ts[i], true);
		}
		choiceUnif = -1;
		max = -2.0;

		for (int j = 0; j < appxSilhouettesUnif.length; j++) {

			errorsUnif[j] = Math.abs(exactSilhouettes[j] - appxSilhouettesUnif[j]);

			if(appxSilhouettesUnif[j] > max) {
				max = appxSilhouettesUnif[j];
				choiceUnif = j;
			}
		}
		log.writeLn("choice Uniform:"+(choiceUnif + 1)+"\n***");
		System.out.println("Uniform: "+ (choiceUnif+1));

		double[] errorsSimple = new double[confs.length];
		
		for (int i = 0; i < errorsSimple.length; i++) {
			errorsSimple[i] = Math.abs(simplifiedSilhouettes[i] - exactSilhouettes[i]);
		}

		for (int i = 0; i < confs.length; i++) {
			log.writeLn("conf "+ (i+1)+ ":\n"
					+ "   Simplified:\n"
					+ "       silhouette: "+simplifiedSilhouettes[i]+"\n"
					+ "       error: "+errorsSimple[i]+"\n"
					+ "   PPS:\n"
					+ "       silhouette: "+appxSilhouettesPPS[i]+"\n"
					+ "       error: "+errorsPPS[i]+"\n"
					+ "   Uniform:\n"
					+ "       silhouette: "+appxSilhouettesUnif[i]+"\n"
					+ "       error: "+errorsUnif[i]+"\n");
		}

		log.close();
		
		System.out.println("Done!");
	}

	private static double correctnessAppx2(Dataset[] conf, double eps, double delta, int runs,
			double exactSilhouette, Logger log, int n, int t, boolean naiveSampling) {

		double[] appxSilhouettes = new double[runs];
		double[] errors = new double[runs];

		Dataset[] samples = new Dataset[conf.length];

		for (int i = 0; i < runs; i++) {

			for (int j = 0; j < samples.length; j++) {
				samples[j] = new DefaultDataset();
			}

			double[][] samplingProbabs = new double[conf.length][];

			if (naiveSampling) {
				samplingProbabs = UniformSampler.sample(conf, samples, t);
			} else {
				double[][] allpr = new double[conf.length][];
				samplingProbabs = PPSSampler.sample(conf, samples, eps, delta, t, allpr);
			}

			/*
			 * silhouette appx
			 */
			SamplingSilhouetteApprox appx = new SamplingSilhouetteApprox();
			appxSilhouettes[i] = appx.score(conf, samples, samplingProbabs);
			errors[i] = Math.abs(appxSilhouettes[i] - exactSilhouette);
		}

		double toret = 0;

		for (int i = 0; i < appxSilhouettes.length; i++) {
			toret = toret + appxSilhouettes[i];
		}

		return toret/(runs*1.0);


	}

	private static void performanceTestNaive(String[] args) throws IOException {

		int index = -1;

		String dataFile = null;
		String logFile = null;
		String csvFile = null;
		double eps;
		double delta;
		boolean append = false;

		/*
		 * arguments parsing
		 */

		if ((index = Utils.index(args, "dataset")) != -1) {
			dataFile = args[index + 1];
		} else {
			throw new IllegalArgumentException("no dataset source");
		}

		if ((index = Utils.index(args, "log")) != -1) {
			logFile = args[index + 1];
		} else {
			logFile = "Experiment_" + System.currentTimeMillis() + ".log";
		}

		if ((index = Utils.index(args, "csv")) != -1) {
			csvFile = args[index + 1];
		} else {
			csvFile = "Experiment_" + System.currentTimeMillis() + "CSV.csv";
		}

		if ((index = Utils.index(args, "epsilon")) != -1) {
			eps = Double.parseDouble(args[index + 1]);
		} else {
			eps = 0.01;
		}

		if ((index = Utils.index(args, "delta")) != -1) {
			delta = Double.parseDouble(args[index + 1]);
		} else {
			delta = 0.01;
		}

		if ((index = Utils.index(args, "append")) != -1) {
			append = true;
		}

		/*
		 * Dataset loading
		 */

		Logger log = new Logger(logFile, false);
		Logger csvStats = new Logger(csvFile, append);

		Dataset[] clusters = parseCSVData(dataFile);

		int n = 0;

		for (int i = 0; i < clusters.length; i++) {
			n = n + clusters[i].size();
		}

		int k = clusters.length;

		log.writeLn("***EXPERIMENT RESUME***");
		log.writeLn("\n");
		log.writeLn("***\n");
		log.writeLn("Total dataset size: " + n + "\n");
		log.writeLn("Absolute error threshold for single element silhouette: " + eps);
		log.writeLn("Probability to have a higher error: " + delta);
		log.write("Sampling probability strategy: Naive");

		log.writeLn("\n");
		log.writeLn("***\n");

		System.out.println("Total dataset size: " + n);

		if (!append) {

			csvStats.write("type,n,k,t,");
			for (int i = 0; i < clusters.length; i++) {
				csvStats.write("n" + (i + 1) + ",s" + (i + 1) + ",");
			}
			csvStats.writeLn("delta,eps,estimatedSilh,sampTime,calcTimes,overallTimes");
		}

		log.writeLn("***\n");

		int t = (int) Math
				.ceil((1 / (2 * Math.pow(eps, 2))) * (Math.log(4) + Math.log(n) + Math.log(k) - Math.log(delta)));

		System.out.println("Sampling phase - Minumum expected sample size: " + t);

		/*
		 * Step 1: sampling phase
		 */

		long startCpuTimeNanoSampling = Utils.getUserTime();

		Dataset[] samples = new Dataset[clusters.length];

		for (int i = 0; i < samples.length; i++) {
			samples[i] = new DefaultDataset();
		}

		double[][] samplingProbabs = new double[clusters.length][];

		samplingProbabs = UniformSampler.sample(clusters, samples, t);

		double samplingTime = ((Utils.getUserTime() - startCpuTimeNanoSampling) / 10e8);

		log.writeLn("Expected sample size: " + t + "\n");

		for (int i = 0; i < samples.length; i++) {
			log.writeLn("Sampling for cluster " + (i + 1) + ":");
			log.writeLn("Original cluster size: " + clusters[i].size());
			log.writeLn("cluster sample size: " + samples[i].size() + "\n");
		}

		/*
		 * Step 2: silhouette
		 */

		SamplingSilhouetteApprox appx = new SamplingSilhouetteApprox();

		long startCpuTimeNanoSilhEval = Utils.getUserTime();

		double silhouette = appx.score(clusters, samples, samplingProbabs);

		double calcTime = (Utils.getUserTime() - startCpuTimeNanoSilhEval) / 10e8;
		double overallTime = (Utils.getUserTime() - startCpuTimeNanoSampling) / 10e8;

		log.writeLn("silhouette calculated with PPS sampling-based algorithm: " + silhouette + "\n");

		log.writeLn("elapsed CPU time Sampling Phase: " + samplingTime + "s");
		log.writeLn("elapsed CPU time for silhouette calculation: " + calcTime + "s");
		log.writeLn("overall elapsed CPU time: " + overallTime + "s");

		System.out.println(" Sampling-based algorithm: " + silhouette);

		csvStats.write("Naive," + n + "," + k + "," + t + ",");
		for (int i = 0; i < clusters.length; i++) {
			csvStats.write(clusters[i].size() + "," + samples[i].size() + ",");
		}
		csvStats.writeLn(
				delta + "," + eps + "," + silhouette + "," + samplingTime + "," + calcTime + "," + overallTime);

		log.writeLn("***");
		log.writeLn("RESUME");
		log.writeLn("***");
		log.writeLn("Times: ");
		log.writeLn("average sampling time: " + samplingTime);
		log.writeLn("average appx silhouette calculation time: " + calcTime);
		log.writeLn("average overall time: " + overallTime + "\n");

		log.writeLn("Measures: ");
		log.writeLn("average approximated silhouette: " + silhouette);

		log.close();

		csvStats.close();

		System.out.println("Done!");

	}

	private static Dataset[] parseCSVData(String dataFile) throws IOException {

		File file = new File(dataFile);
		Scanner sc = new Scanner(file);

		sc.nextLine();
		String infos = sc.nextLine();

		StringTokenizer tokenizer = new StringTokenizer(infos, ",");

		int dimensions = Integer.parseInt(tokenizer.nextToken());
		int k = Integer.parseInt(tokenizer.nextToken());

		Dataset[] clusters = new Dataset[k];

		for (int i = 0; i < clusters.length; i++) {
			clusters[i] = new DefaultDataset();
		}

		while (sc.hasNextLine()) {
			tokenizer = new StringTokenizer(sc.nextLine(), ",");
			double[] elements = new double[dimensions];
			for (int i = 0; i < dimensions; i++) {
				elements[i] = Double.parseDouble(tokenizer.nextToken());
			}
			clusters[Integer.parseInt(tokenizer.nextToken())].add(new DenseInstance(elements));
		}

		sc.close();

		return clusters;

	}

}
