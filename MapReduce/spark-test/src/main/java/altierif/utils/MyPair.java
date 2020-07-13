package altierif.utils;

import java.io.Serializable;

import org.apache.spark.ml.linalg.Vector;

public class MyPair implements Serializable{
	public Vector feats;
	public long clust;
	
	public Vector getFeats() {
		return feats;
	}
	public void setFeats(Vector feats) {
		this.feats = feats;
	}
	public long getClust() {
		return clust;
	}
	public void setClust(long clust) {
		this.clust = clust;
	}
	public MyPair(Vector feats, long clust) {
		super();
		this.feats = feats;
		this.clust = clust;
	}
}
