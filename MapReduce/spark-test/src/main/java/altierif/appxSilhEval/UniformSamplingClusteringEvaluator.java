package altierif.appxSilhEval;

import java.util.Iterator;
import java.util.List;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.api.java.function.ReduceFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.evaluation.Evaluator;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.ParamPair;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import altierif.utils.MyPair;
import altierif.utils.SampleAccumulator;
import altierif.utils.SamplePair;
import scala.collection.JavaConversions;

/**
 * @author Federico Altieri
 * 
 * Class that implements the Spark implementation of the Map-Reduce algorithm for uniform-sampling silhouette approximation
 * with euclidean distance
 *
 */
@SuppressWarnings("serial")
public class UniformSamplingClusteringEvaluator extends Evaluator {

	private String uid;
	private double delta;
	private double epsilon;
	private int t;
	private int partitions;
	public static final double DEFAULT_DELTA = 0.1;
	public static final double DEFAULT_EPS = 0.1;

	public UniformSamplingClusteringEvaluator() {
		this("UnifShilhEval" + Long.toString(System.currentTimeMillis()));
	}

	/**
	 * Instantiates a new evaluator instance with the parameters of delta and
	 * epsilon passed as arguments (and a random UID).
	 * 
	 * @param delta
	 * @param epsilon
	 */
	public UniformSamplingClusteringEvaluator(double delta, double epsilon) {
		this("UnifShilhEval" + Long.toString(System.currentTimeMillis()), delta, epsilon, 0);
	}

	public UniformSamplingClusteringEvaluator(int t) {
		this("UnifShilhEval" + Long.toString(System.currentTimeMillis()), DEFAULT_DELTA, DEFAULT_EPS, t);
	}

	/**
	 * Instantiates a new evaluator instance with the parameters of delta and
	 * epsilon. set both to default value 0.1
	 * 
	 * @param uid objects's UID
	 */
	public UniformSamplingClusteringEvaluator(String uid) {
		this(uid, DEFAULT_DELTA, DEFAULT_EPS, 0);
	}


	/**
	 * Instantiates a new evaluator instance with the parameters of delta and
	 * epsilon passed as arguments and given UID.
	 * 
	 * @param uid   object's UID to set
	 * @param delta
	 * @param eps
	 */
	public UniformSamplingClusteringEvaluator(String uid, double delta, double eps, int t) {
		super();
		this.uid = uid;
		this.delta = delta;
		this.epsilon = eps;
		this.t = t;
	}
	
	public void setPartitions(int partitions) {
		this.partitions = partitions;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String uid() {
		return uid;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public Evaluator copy(ParamMap extra) {
		PPSSilhouetteEstimtor toReturn = new PPSSilhouetteEstimtor(uid);
		List<ParamPair<?>> params = extra.toList();
		for (ParamPair<?> paramPair : params) {
			toReturn.set(paramPair);
		}
		return toReturn;
	}

	/**
	 * Performs silhouette approximation in a 5 steps Map-Reduce algorithm.
	 *
	 * @param {@link Dataset} instance to be processed.
	 * @return double value of the approximated silhouette value.
	 */

	@Override
	public double evaluate(Dataset<?> dataset) {
		
		@SuppressWarnings("unchecked")
		Dataset<Row> coll = (Dataset<Row>)dataset;

		double all = coll.count();

		List<Row> infos = coll.groupBy("prediction").count().collectAsList();

		SparkSession spark = SparkSession.getActiveSession().get();

		JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());

		coll.printSchema();
		if (this.partitions <= 0) {
			this.partitions = 1;
		}
		Column[] columnArray = new Column[1];
		columnArray[0] = new Column("index");

		coll = coll.repartition(this.partitions).cache();

		int k = infos.size();



		System.out.println("n:" + all);
		System.out.println("k:" + k);
		System.out.println("t:" + t);
		System.out.println("partitions:" + this.partitions);

		Broadcast<int[]> dataInfo = sc.broadcast(new int[] { k, t });

		double[] clustersSizes = new double[infos.size()];

		for (Iterator<Row> iterator = infos.iterator(); iterator.hasNext();) {
			Row row = (Row) iterator.next();
			clustersSizes[((Long)row.getAs("prediction")).intValue()] = ((Long)(row.getAs("count"))).doubleValue();
		}

		Broadcast<double[]> clustSizes = sc.broadcast(clustersSizes);

		if (this.t == 0) {
			t = (int) Math.ceil((1/(2*Math.pow(epsilon, 2)))*(Math.log(4)+Math.log(all)+Math.log(k)-Math.log(delta)));
		}

		SampleAccumulator initsamples = new SampleAccumulator(k);

		JavaSparkContext.toSparkContext(sc).register(initsamples);

		SampleAccumulator samples = new SampleAccumulator(k);

		JavaSparkContext.toSparkContext(sc).register(samples);

		/**PHASE 1**/

		/**
		 * Unpacking of scala vector
		 **/

		Dataset<MyPair> coll2 = coll.map((MapFunction<Row, MyPair>) row -> 
		{	
			Object[] temp;
			double[] pivotarr;

			temp = JavaConversions.mutableSeqAsJavaList(((Row)(row.getAs("features"))).getAs("values")).toArray();
			pivotarr = new double[temp.length];
			for (int i = 0; i < temp.length; i++) {
				pivotarr[i] = ((Number)temp[i]).doubleValue();
			}

			return new MyPair(Vectors.dense(pivotarr), row.getAs("prediction"));
		}
		, 
		Encoders.javaSerialization(MyPair.class)
				);


//		coll2.foreach((ForeachFunction<MyPair>) row ->{
//
//			int cluster = Long.valueOf(row.getClust()).intValue();
//
//			double clustSize = clustSizes.value()[cluster];
//
//
//			if (clustSize > t) {
//				if (Math.random() < ((2/clustSize)*s0.value().doubleValue()))
//				{
//					initsamples.add(new SamplePair(row.getFeats(), 1), cluster);
//				}
//			} else {
//				samples.add(new SamplePair(row.getFeats(), 1.0), cluster);
//			}
//		});
//
//		int maxSize = 0;
//
//		for (int i = 0; i < k; i++) {
//			maxSize = Math.max(initsamples.valueAt(i).size(), maxSize);
//		}
//
//
//		DistAccumulator distSums = new DistAccumulator(k, maxSize);
//
//		JavaSparkContext.toSparkContext(sc).register(distSums);
//
//		System.out.println("maxsize "+maxSize);
//		
//		Broadcast<List<SamplePair>[]> initsam = sc.broadcast(initsamples.value());

//		/**
//		 * Computes the distance sums from initial samples
//		 * 
//		 * **/
//
//
//		coll2.foreach((ForeachFunction<MyPair>) elem ->{
//
//			int clust = Long.valueOf(elem.getClust()).intValue();
//			Iterator<SamplePair> iterSam = initsam.value()[clust].iterator();
//			int ct = 0;
//			while (iterSam.hasNext()) {
//				distSums.add(Math.sqrt(Vectors.sqdist(elem.getFeats(), iterSam.next().getFeats())), clust, ct);
//				ct++;
//			}
//
//
//		});

		/**
		 * MAP PHASE 2: compute probabilities and collect the PPS sample
		 * **/
		
//		Broadcast<double[][]> distances = sc.broadcast(distSums.value());


		coll2.foreach((ForeachFunction<MyPair>) pivot ->{

			int clust = Long.valueOf(pivot.getClust()).intValue();
			double probability = dataInfo.getValue()[1]/clustSizes.value()[clust];
			
			if (Math.random() < probability)
				samples.add(new SamplePair(pivot.getFeats(), probability), clust);

		});


		/**
		 * REDUCE PHASE 3: computes the pointwise silhouettes value with the sampled points and sums them
		 * 
		 * **/
		
		Broadcast<List<SamplePair>[]> PPSsam = sc.broadcast(samples.value());

		return coll2.map((MapFunction<MyPair, Double>) pivot ->{

			int clust = Long.valueOf(pivot.getClust()).intValue();
			double a = 0.0;
			double b = Double.MAX_VALUE;
			List<SamplePair>[] list = PPSsam.value();
			for (int i = 0; i < list.length; i++) {
				Iterator<SamplePair> samp = list[i].iterator();
				if (i == clust) {
					while (samp.hasNext()) {
						SamplePair s = samp.next();
						a = a + (Math.sqrt(Vectors.sqdist(pivot.getFeats(), s.getFeats()))/s.getProbability());
					}
					a = a/(clustSizes.value()[i]-1.0);
				}
				else{
					double temp = 0.0;
					while (samp.hasNext()) {
						SamplePair s = samp.next();
						temp = temp + (Math.sqrt(Vectors.sqdist(pivot.getFeats(), s.getFeats()))/s.getProbability());
					}
					b = Double.min(b, (temp/clustSizes.value()[i]));
				}
			}
			return Double.valueOf((b-a)/Double.max(a, b));
		},
				Encoders.DOUBLE())
		
		/** sums all the silhouette sums into an only one**/		
				
		.reduce((ReduceFunction<Double>) (s1,s2) ->{
			return (s1.doubleValue() + s2.doubleValue());
		}).doubleValue()/all;

	}

}
