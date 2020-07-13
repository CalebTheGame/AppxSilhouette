package altierif.utils;

import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.util.AccumulatorV2;

public class CentroidAccumulator extends AccumulatorV2<DenseVector[], DenseVector[]> {

	private DenseVector[] centroids;
	private boolean isZero;

	public CentroidAccumulator(int k, int dimensions) {
		super();
		centroids = new DenseVector[k];
		for (int i = 0; i < k; i++) {
			centroids[i] = Vectors.zeros(dimensions).toDense();
		}
		isZero = true;
	}

	public CentroidAccumulator(DenseVector[] centroids) {
		super();
		this.centroids = centroids;
		isZero = true;
	}

	@Override
	public void add(DenseVector[] v) {
		DenseVector[] out = new DenseVector[centroids.length];
		for (int i = 0; i < centroids.length; i++) {
			double[] base = centroids[i].toArray();
			double[] toSum = v[i].toArray();
			for (int j = 0; j < base.length; j++) {
				base[j] = base[j] + toSum[j];
			}
			out[i] = Vectors.dense(base).toDense();
		}
		centroids = out;
		isZero = false;
	}

	public void add(DenseVector v, int cluster) {
		double[] base = centroids[cluster].toArray();
		double[] toSum = v.toArray();
		for (int j = 0; j < base.length; j++) {
			base[j] = base[j] + toSum[j];
		}
		centroids[cluster] = Vectors.dense(base).toDense();
		isZero = false;
	}

	@Override
	public AccumulatorV2<DenseVector[], DenseVector[]> copy() {
		return new CentroidAccumulator(centroids);
	}

	@Override
	public boolean isZero() {
		if (isZero) 
			return true;
		else 
			return false;
	}

	@Override
	public void merge(AccumulatorV2<DenseVector[], DenseVector[]> other) {
		this.add(other.value());
		isZero = false;
	}

	@Override
	public void reset() {
		int dimensions = centroids[0].size();
		centroids = new DenseVector[centroids.length];
		for (int i = 0; i < centroids.length; i++) {
			centroids[i] = Vectors.zeros(dimensions).toDense();
		}
		isZero = true;
	}

	@Override
	public DenseVector[] value() {
		return centroids;
	}

}
