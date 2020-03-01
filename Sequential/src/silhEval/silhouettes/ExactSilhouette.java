package silhEval.silhouettes;

import java.util.Iterator;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.distance.EuclideanDistance;

/**
 * Calculator of the exact silhouette calculation with a sequential algorithm. The silhouette is computed according to Rousseew's definition. (see 
 * <a href="https://doi.org/10.1016/0377-0427(87)90125-7">Silhouettes: a graphical aid to the interpretation and validation of cluster analysis</a>).
 * 
 * Limits: dataset size is limited by limitations in int value
 * 
 * @author Federico Altieri
 * 
 *
 */
public class ExactSilhouette implements Silhouette {
	/**
	 * Array with all the scores of all elements.
	 */
	private double[][] scores;
	
	private boolean computed = false;
	
	public boolean isComputed() {
		return computed;
	}

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
	public double score(Dataset[] arg0) {
		if (!computed) {
			computeScores(arg0);
		}
		return overallScore();
	}

	/**
	 * Computes the silhouette value for all the dataset (obtained as the average of the silhouette value of the single elements).  
	 * 
	 * @param Clusters Array of {@link Dataset} instances representing the dataset partitioned into clusters.
	 * @return The value of the silhouette.
	 */
	private double overallScore() {
		
		double silhSums = 0;
		int size = 0;
		
		for (int i = 0; i < scores.length; i++) {
			size = size + scores[i].length;
			for (int j = 0; j < scores[i].length; j++) {
				silhSums = silhSums + scores[i][j]; 
			}
		}
	
		return silhSums/(size);
	}
	
	/**
	 * Computes scores of all elements in the dataset
	 * 
	 * @param clusters Array of {@link Dataset} instances representing the dataset partitioned into clusters.
	 */
	private void computeScores(Dataset[] clusters) {
		scores = new double[clusters.length][];
		int size = 0;
		double silhSums = 0;
		for (int i = 0; i < clusters.length; i++) {
			// accumulates the size to get the whole dataset size
			size = size + clusters[i].size();
			scores[i] = new double[clusters[i].size()];
			//iterator to get all samples
			Iterator<Instance> iter = clusters[i].iterator();
			int ct = 0;
			while (iter.hasNext()) {
				Instance current = (Instance) iter.next();
				scores[i][ct] = getSingleElementScore(current, clusters, i);
				silhSums = silhSums + scores[i][ct];
				ct++;
			}
		}
		computed = true;
	}
	
	/**
	 * Computes the sum of the Euclidean distances between an element (instance of {@link Instance}) and all element of a non clustered dataset ({@link Dataset} instance).
	 * Typical use is to compute distances sum of a single element with a single cluster of a clustered dataset. 
	 * 
	 * @param source {@link Instance} instance representing the element.
	 * @param target {@link Dataset} of the elements to compute distances. 
	 * @return The sum of the Euclidean distances.
	 */
	private static double getDatasetDistanceSum (Instance source, Dataset target) {
		Iterator<Instance> iterTgt = target.iterator();
		EuclideanDistance calc = new EuclideanDistance();
		double sum = 0;
		while (iterTgt.hasNext()) {
			Instance tgt = (Instance) iterTgt.next();
			sum = sum + calc.calculateDistance(source, tgt);
		}
		return sum;		
	}
	
	/**
	 * Computes the naive silhouette value for a single element of a dataset.
	 * 
	 * @param element The element to compute the silhouette.
	 * @param clusters Array of {@link Dataset} instances representing the dataset partitioned into clusters.
	 * @param clustIndex Index of the cluster where the element is.
	 * @return The silhouette value for the element.
	 */
	public double getSingleElementScore(Instance element, Dataset[] clusters, int clustIndex) {
		double a;
		if (clusters[clustIndex].size() == 1) {
			a = 0.0;
		} else {
			a = getDatasetDistanceSum(element, clusters[clustIndex]) / (double)(clusters[clustIndex].size()-1);
		}
		double b = Double.MAX_VALUE;
		for (int j = 0; j < clusters.length; j++) {
			if (clustIndex != j) {
				double candidate = getDatasetDistanceSum(element, clusters[j]) / clusters[j].size();
				if (b > candidate) {
					b = candidate;
				}
			}
		}
		return ((b - a) / Math.max(a, b));
	}

	/**
	 *{@inheritDoc}
	 */
	@Override
	public double[][] getScores(Dataset[] clusters) {
		if (!computed) {
			computeScores(clusters);
		}
		return scores;
	}

}
