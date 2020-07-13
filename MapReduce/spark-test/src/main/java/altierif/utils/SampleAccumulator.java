package altierif.utils;

import java.util.LinkedList;
import java.util.List;

import org.apache.spark.util.AccumulatorV2;

public class SampleAccumulator extends AccumulatorV2<List<SamplePair>[], List<SamplePair>[]>{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	List<SamplePair>[] samples;
	boolean zero;

	public SampleAccumulator(List<SamplePair>[] samples) {
		super();
		this.samples = samples;
		zero = false;
	}

	public SampleAccumulator(int clusters) {
		super();
		this.samples = new LinkedList[clusters];
		this.reset();
	}

	@Override
	public void add(List<SamplePair>[] v) {
		for (int i = 0; i < v.length; i++) {
			List<SamplePair> toAdd = v[i];
			for (SamplePair dv : toAdd) {
				samples[i].add(dv);
			}
		}
		zero = false;
	}

	public void add(SamplePair v, int index) {
		samples[index].add(v);
		zero = false;
	}

	@Override
	public AccumulatorV2<List<SamplePair>[], List<SamplePair>[]> copy() {
		return new SampleAccumulator(samples);
	}

	@Override
	public boolean isZero() {
		if(samples == null)
			return true;
		else if(samples[0] == null) {
			return true;
		}
		return zero;
	}

	@Override
	public void merge(AccumulatorV2<List<SamplePair>[], List<SamplePair>[]> other) {
		this.add(other.value());
	}

	@SuppressWarnings("unchecked")
	@Override
	public void reset() {
		samples = new LinkedList[samples.length];
		for (int i = 0; i < samples.length; i++) {
			samples[i] = new LinkedList<SamplePair>();
		}
		zero = true;
	}

	@Override
	public List<SamplePair>[] value() {
		return samples;
	}

	public List<SamplePair> valueAt(int index) {
		return samples[index];
	}


}
