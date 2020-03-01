package silhEval.silhouettes;

import java.util.Iterator;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.distance.EuclideanDistance;
import net.sf.javaml.tools.DatasetTools;

/**
 * Calculator of the Simplified Silhouette with a sequential algorithm. 
 * Limits: dataset size is limited by limitations in int value.
 * 
 * 
 * @author Federico Altieri
 * 
 *
 */
public class SimplifiedSilhouette implements Silhouette {
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
	 * @param Clusters array of {@link Dataset} instances representing the dataset partitioned into clusters.
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
	 * @param clusters
	 */
	private void computeScores(Dataset[] clusters) {
		scores = new double[clusters.length][];
		int size = 0;
		double silhSums = 0;
		Instance[] representers = new Instance[clusters.length];
		for (int i = 0; i < representers.length; i++) {
			Instance centroid = DatasetTools.average(clusters[i]);
			representers[i] = clusters[i].kNearest(1, centroid, new EuclideanDistance()).iterator().next();
		}
		for (int i = 0; i < clusters.length; i++) {
			// accumulates the size to get the whole dataset size
			size = size + clusters[i].size();
			scores[i] = new double[clusters[i].size()];
			//iterator to get all samples
			Iterator<Instance> iter = clusters[i].iterator();
			int ct = 0;
			while (iter.hasNext()) {
				Instance current = (Instance) iter.next();
				scores[i][ct] = getSingleElementScore(current, representers, i);
				silhSums = silhSums + scores[i][ct];
				ct++;
			}
		}
		computed = true;
	}
	
	/**
	 * Computes the naive silhouette value for a single element of a dataset.
	 * 
	 * @param element The element to compute the silhouette.
	 * @param clusters Array of {@link Dataset} instances representing the dataset partitioned into clusters.
	 * @param clustIndex Index of the cluster where the element is.
	 * @return The silhouette value for the element.
	 */
	public double getSingleElementScore(Instance element, Instance[] representers, int clustIndex) {
		EuclideanDistance calc = new EuclideanDistance();
		double a;
		a = calc.calculateDistance(element, representers[clustIndex]);
		double b = Double.MAX_VALUE;
		for (int j = 0; j < representers.length; j++) {
			if (clustIndex != j) {
				double candidate = calc.calculateDistance(element, representers[j]);
				if (b > candidate) {
					b = candidate;
				}
			}
		}
		return ((b - a) / Math.max(a, b));
	}

	@Override
	public double[][] getScores(Dataset[] clusters) {
		if (!computed) {
			computeScores(clusters);
		}
		return scores;
	}

	@Override
	public double getSingleElementScore(Instance element, Dataset[] clusters, int clustIndex) {
		Instance[] representers = new Instance[clusters.length];
		for (int i = 0; i < representers.length; i++) {
			Instance centroid = DatasetTools.average(clusters[i]);
			representers[i] = clusters[i].kNearest(1, centroid, new EuclideanDistance()).iterator().next();
		}
		return getSingleElementScore(element, representers, clustIndex);
	}

}
