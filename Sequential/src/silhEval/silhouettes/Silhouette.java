package silhEval.silhouettes;

import net.sf.javaml.clustering.evaluation.ClusterEvaluation;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;

/**
 * Abstract class that represent the concept of silhouette, according to to Rousseew's definition. (see 
 * <a href="https://doi.org/10.1016/0377-0427(87)90125-7">Silhouettes: a graphical aid to the interpretation and validation of cluster analysis</a>).
 *  
 * 
 * @author Federico Altieri
 *
 */
public interface Silhouette extends ClusterEvaluation {
	
	/**
	 * Returns the silhouette value of a single element within a clusterized dataset.
	 * 
	 * @param element The element to compute the score. 
	 * @param clusters CLusterized dataset to compute 
	 * @param clustIndex Index of the cluster where the element is
	 * @return The silhouette value.
	 */
	public double getSingleElementScore(Instance element, Dataset[] clusters, int clustIndex);
	
	/**
	 * Returns the scores of all elements
	 * 
	 * @param clusters
	 * @return
	 */
	public double[][] getScores(Dataset[] clusters);


}
