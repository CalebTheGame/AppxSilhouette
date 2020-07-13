package altierif.utils;

import java.io.Serializable;

import org.apache.spark.ml.linalg.Vector;

public class SamplePair implements Serializable{
	public Vector feats;
	public double probab;

	public Vector getFeats() {
		return feats;
	}
	public void setFeats(Vector feats) {
		this.feats = feats;
	}
	public double getProbability() {
		return probab;
	}
	public void setClust(double clust) {
		this.probab = clust;
	}
	public SamplePair(Vector feats, double probab) {
		super();
		this.feats = feats;
		this.probab = probab;
	}
}
