package silhEval.silhouettes;

import net.sf.javaml.core.Dataset;

/**
 * 
 * This interface provides a frame for all sampling based approximated measure that can be used to evaluate
 * the quality of a cluster over a dataset.
 * 
 * @author Federico Altieri
 *
 */
public interface SamplingBasedClusterEvaluation {
	
	/**
	 * returns the silhouette approximated through a sampling
	 * 
	 * @param clusters array of {@link Dataset} instances representing the dataset partitioned into clusters.
	 * @param sample Array of {@link Dataset} instances containing the union of clusters samples.
	 * @param probabs Two-dimensioned array of double that contains all the probabilities of the samples.
	 * @return the approximate silhouette score of the clusterized dataset
	 */
	
	public double score(Dataset[] clusters, Dataset[] sample, double[][]probabs);
	
	/**
     * Compares the two scores according to the criterion in the implementation.
     * Some score should be maxed, others should be minimized. This method
     * returns true if the second score is 'better' than the first score.
     * 
     * @param score1
     *            the first score
     * @param score2
     *            the second score
     * @return true if the second score is better than the first, false in all
     *         other cases
     */
    public boolean compareScore(double score1, double score2);

}
