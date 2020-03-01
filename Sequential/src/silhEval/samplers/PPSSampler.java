package silhEval.samplers;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.distance.EuclideanDistance;

/**
 * Container of static methods that implement a sampling that uses the "Probability Proportional to Size" definition by Chechik, Cohen and Kaplan in <a href="https://arxiv.org/abs/1503.08528">Average Distance Queries through Weighted Samples in Graphs and Metric Spaces: 
 * High Scalability with Tight Statistical Guarantees</a>. PPS sampling is exploited here to obtain a collection of samples of a clustered dataset that has precise high probability
 * guarantees on the error of distance sums of each element of the metric space of a dataset.
 * 
 * @author Federico Altieri
 *
 */
public class PPSSampler {

	/**
	 * Performs the sampling of a clustered dataset, each one sampled according to PPS sampling.
	 * 
	 * @param clusters Clusters Array of {@link Dataset} instances representing the dataset partitioned into clusters.
	 * @param sample Array of {@link Dataset} instances that will contain the union of clusters samples.
	 * @param eps Maximum absolute error threshold for the distance sums of any element in space respect to each cluster. 
	 * @param delta Probability that the error between the estimated distance sum and the effective distance sum exceeds the error threshold passed in parameter "eps".
	 * @param t Expected size of each sample.
	 * @param allpr Two-dimensioned array of double that will contain all the probabilities of every element of the whole clustered data.
	 * @return the Two-dimensioned array of double that contains the probabilities of the samples.
	 */
	public static double[][] sample(Dataset[] clusters, Dataset[] sample, double eps, double delta, int t, double[][] allpr) {
		
		int n = 0;
		for (int i = 0; i < clusters.length; i++) {
			allpr[i] = new double[clusters[i].size()];
			n = n + clusters[i].size();
		}
		
		double[][] probabs = new double[clusters.length][];
		for (int i = 0; i < clusters.length; i++) {
			probabs[i] = sampleSingleCluster (clusters[i], sample[i], eps, delta, n, clusters.length, t, allpr[i]);
		}
		return probabs;
	}

	/**
	 * Samples a single cluster according to PPS sampling.
	 * 
	 * @param cluster Array of {@link Dataset} instances representing the cluster to sample. 
	 * @return The sample of the cluster.
	 */
	/**
	 * Samples a single cluster according to PPS sampling.
	 * 
	 * @param cluster Array of {@link Dataset} instances representing the cluster to sample.
	 * @param sample Array of {@link Dataset} instances that will contain the sample of the cluster.
	 * @param eps Maximum absolute error threshold for the distance sum of any element in space with the cluster.
	 * @param delta Probability that the error between the estimated distance sum and the effective distance sum exceeds the error threshold passed in parameter "eps".
	 * @param n Total number of all elements in the dataset.
	 * @param k Number of clusters.
	 * @param t Expected size of each sample.
	 * @param probabs Array of double that will contain all the probabilities of every element of the cluster.
	 * @return Array of double that contains the probabilities of the samples.
	 */
	private static double[] sampleSingleCluster(Dataset cluster, Dataset sample, double eps, double delta, int n, int k, int t, double[] probabs) {
		
		//int t = (int)Math.ceil((1/(2*Math.pow(eps, 2)))*(Math.log(4)+Math.log(n)+Math.log(k)-Math.log(delta)));
		
		if (t > cluster.size()) {
			for (Iterator<Instance> iterator = cluster.iterator(); iterator.hasNext();) {
				//probabs = new double[cluster.size()];
				sample.add((Instance) iterator.next());				
			}			
			Arrays.fill(probabs, 1.0);
			return probabs;
		}
		
		System.arraycopy(getPPSprobabilities(cluster, t, (int) Math.ceil(silhEval.utils.Utils.log2((2*k)/delta))),
				0, probabs, 0, probabs.length);
		
//		probabs = getPPSprobabilities(cluster, t, (int) Math.ceil(silhEval.utils.Utils.log2((2*k)/delta)));
		
		/*
		 * Poisson sampling according to computed coefficients
		 * */		
		int[] indexes = new int[cluster.size()];
		
		int ct =0;
		for (int i = 0; i < probabs.length; i++) {
			if (Math.random() < probabs[i]) {
				indexes[ct] = i;
				sample.add(cluster.get(i));
				ct++;
			}
		}
		
		double toReturn[] = new double[sample.size()];
		
		for (int i = 0; i < toReturn.length; i++) {
			toReturn[i] = probabs[indexes[i]];
		}
		
				
		return toReturn;
	}

	/**
	 * Computes the distance sum of a single element from a set of elements represented by a {@link Dataset} instance.
	 * 
	 * @param element The source element to compute the distance sum.
	 * @param set The set of elements to compute the distances.
	 * @return The value of the distance sum.
	 */
	private static double distanceSum(Instance element, Dataset set) {
		EuclideanDistance calc = new EuclideanDistance();
		double sum = 0.0;
		Iterator<Instance> iter = set.listIterator();
		while (iter.hasNext()) {
			sum = sum + calc.calculateDistance(element, (Instance) iter.next());			
		}
		return sum;
	}
	
	/**
	 * Calculates the array of sampling probabilities according to PPS sampling, with an expected sample size. 
	 * 
	 * @param dataset The {@link Dataset} instance representing data to sample.
	 * @param sampleSize The expected sample size.
	 * @param initialSize The size of the initial sample.
	 * @return the sampling probabilities, ordered according to elements in passed {@link Dataset} instance.
	 */
	private static double[] getPPSprobabilities(Dataset dataset, int sampleSize, int initialSize) {
		/*
		 * collect initial sample S0
		 * */
		
		Dataset temp = dataset.copy();
		
		Dataset initialSample = new DefaultDataset();
		
//		System.out.println("initial sample size:"+initialSize);
		
		Random rand = new Random();
				
		for (int i = 0; i < initialSize; i++) {
			
			initialSample.add(temp.remove((Math.round(rand.nextFloat()*(temp.size()-1)))));
		}
		
		
		/*
		 * initial coefficients gamma
		 * */
		
		double gamma[] = new double[dataset.size()];
		
		Arrays.fill(gamma, (1.0/(double)dataset.size()));
		
		/*
		 * coefficient computation
		 * */
		
		Iterator<Instance> iterS0 = initialSample.listIterator();
		
		EuclideanDistance calc = new EuclideanDistance();
		
		while (iterS0.hasNext()) {
			Instance elementS0 = (Instance) iterS0.next();
			double w = distanceSum(elementS0, dataset);
			for (int i = 0; i < gamma.length; i++) {
				gamma[i] = Math.max(gamma[i], (calc.calculateDistance(elementS0, dataset.get(i)))/w);
			}
		}
		

		/*
		 * calculation of sampling probabilities
		 * */
		
		for (int i = 0; i < gamma.length; i++) {
			gamma[i] = Math.min(gamma[i]*sampleSize, 1);
		}
		
		return gamma;
	}

}
