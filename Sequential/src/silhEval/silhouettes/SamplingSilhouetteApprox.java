package silhEval.silhouettes;

import java.util.Iterator;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.distance.EuclideanDistance;

/**
 * This class calculates the approximated Silhouette score according to a Sampling
 * 
 * @author Federico Altieri
 *
 */
public class SamplingSilhouetteApprox implements SamplingBasedClusterEvaluation{

	/**
	 *{@inheritDoc}
	 */
	@Override
	public boolean compareScore(double arg0, double arg1) {
		return (arg1 > arg0);
	}

	/**
	 *{@inheritDoc}
	 */
	@Override
	public double score(Dataset[] clusters, Dataset[] sample, double[][]probabs) {
		int size = 0;
		double silhSums = 0;
		int[] sizes = new int[clusters.length];
		for (int i = 0; i < sizes.length; i++) {
			sizes[i] = clusters[i].size();
		}
		for (int i = 0; i < clusters.length; i++) {
			// accumulates the size to get the whole dataset size
			size = size + clusters[i].size();
			Iterator<Instance> iter = clusters[i].iterator();
			while (iter.hasNext()) {
				Instance current = (Instance) iter.next();
				silhSums = silhSums + getSingleElementScore(current, sample, i, probabs, sizes);
			}
		}
		
		return silhSums/(size);
	}

	/**
	 * returns the silhouette of a single element approximated through a sampling
	 * 
	 * @param element {@link Instance} to calculate approximated silhouette.
	 * @param sample Array of {@link Dataset} instances containing the union of clusters samples.
	 * @param clustIndex Index of the cluster where the element is.
	 * @param pv Two-dimensioned array of double that contains all the probabilities of the samples.
	 * @param sizes Sizes of the clusters.
	 * @return the approximated silhouette score according to a sample.
	 */
	private double getSingleElementScore(Instance element, Dataset[] sample, int clustIndex, double[][] pv, int[] sizes) {
		//double a = getSampleDistanceSum(element, sample[clustIndex], pv) / (sample[clustIndex].size()-1);
		double a;
		if (sample[clustIndex].size() == 1) {
			a = 0.0;
		}else {
			a = getApproxDistanceSum(element, sample[clustIndex], pv[clustIndex]) / (sizes[clustIndex]-1);
		}
		double b = Double.MAX_VALUE;
		for (int j = 0; j < sample.length; j++) {
			if (clustIndex != j) {
				double candidate = getApproxDistanceSum(element, sample[j], pv[j]) / sizes[j];
				if (b > candidate) {
					b = candidate;
				}
			}
		}
		return ((b - a) / Math.max(a, b));
	}

	/**
	 * Obtain the approximation of the distance sum for an element respect to a collection of samples
	 * 
	 * @param source {@link Instance} to approximate the distance sum.
	 * @param target {@link Dataset} with the samples to use for approximations
	 * @param pv Array of double that contains all the probabilities of the samples.
	 * @return The approximation of the sum
	 */
	private double getApproxDistanceSum(Instance source, Dataset target, double[] pv) {
		EuclideanDistance calc = new EuclideanDistance();
		double sum = 0;
		for (int i = 0; i < pv.length; i++) {
			Instance tgt = target.get(i);
			sum = sum + (calc.calculateDistance(source, tgt)/pv[i]);
		}
		return sum;	
	}

}
