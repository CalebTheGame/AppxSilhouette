package silhEval.samplers;

import java.util.Arrays;
import java.util.Iterator;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;

/**
 * Class that contains static methods that perform a uniform sampling based on uniform probabilities probabilities.
 * Let N the size of the cluster to sample and t the expected sample size, we define "uniform sampling" the sampling of each element independently from each other with probability t/N.
 * 
 * 
 * @author Federico Altieri
 *
 */

public class UniformSampler {

	/**
	 * Perform the sampling of a clustered dataset, each cluster sampled according to uniform sampling.
	 * 
	 * @param clusters Array of {@link Dataset} instances representing the dataset partitioned into clusters.
	 * @param clusters Array of {@link Dataset} instances that will contain the samples of each cluster.
	 * @param t Expected sample size.
	 * @return Two-dimension array of double representing the probabilities of the samples collected.
	 */
	public static double[][] sample(Dataset[] clusters, Dataset[] samples, int t) {

		double[][] probabs = new double[clusters.length][];
		
		for (int i = 0; i < clusters.length; i++) {
			probabs[i] = sampleSingleCluster (clusters[i], samples[i], clusters.length, t);
		}
		return probabs;
	}

	/**
	 * Samples a single cluster according to uniform sampling.
	 * 
	 * @param cluster {@link Dataset} instance representing the cluster to sample.
	 * @param cluster {@link Dataset} instance representing hat will contain the sample of the cluster.
	 * @param k Number of clusters.
	 * @param t Expected sample size.
	 * @return Array of double representing the probabilities of the samples collected.
	 */
	private static double[] sampleSingleCluster(Dataset cluster, Dataset sample, int k, int t) {
		
		double probabs [] = new double[cluster.size()];

		if (t >= cluster.size()) {
			for (Iterator<Instance> iterator = cluster.iterator(); iterator.hasNext();) {
				sample.add((Instance) iterator.next());				
			}
			Arrays.fill(probabs, 1.0);
			return probabs;
		}

		Arrays.fill(probabs, (double)t/(double)cluster.size());
		

		/*
		 * sampling according to computed coefficients
		 * */


		for (int i = 0; i < cluster.size(); i++) {
			
			if (Math.random() <= probabs[i]) {
				sample.add(cluster.get(i));
			}
			
		}
		
		probabs = new double[sample.size()];
		
		Arrays.fill(probabs, (double)t/(double)cluster.size());

		return probabs;
	}

}
