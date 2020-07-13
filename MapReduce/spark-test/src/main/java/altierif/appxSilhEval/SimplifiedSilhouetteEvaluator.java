package altierif.appxSilhEval;


import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.api.java.function.MapPartitionsFunction;
import org.apache.spark.api.java.function.ReduceFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.evaluation.Evaluator;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.ParamPair;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.AccumulatorV2;
import altierif.appxSilhEval.PPSSilhouetteEstimtor;
import altierif.utils.CentroidAccumulator;
import altierif.utils.MyPair;
import altierif.utils.SamplePair;
import scala.collection.JavaConversions;

/**
 * @author Federico Altieri
 * 
 *         Class that implements the Spark implementation of the Map-Reduce
 *         algorithm for simplified silhouette computation with euclidean
 *         distance
 *
 */
@SuppressWarnings("serial")
public class SimplifiedSilhouetteEvaluator extends Evaluator {

	private String uid;
	private int partitions;

	public SimplifiedSilhouetteEvaluator() {
		this("ExactShilhEval" + Long.toString(System.currentTimeMillis()));
	}

	/**
	 * Instantiates a new evaluator instance with the parameters of delta and
	 * epsilon. set both to default value 0.1
	 * 
	 * @param uid objects's UID
	 */
	public SimplifiedSilhouetteEvaluator(String uid) {
		super();
		this.uid = uid;
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
		Dataset<Row> coll = (Dataset<Row>) dataset;

		double all = coll.count();

		List<Row> infos = coll.groupBy("prediction").count().collectAsList();

		SparkSession spark = SparkSession.getActiveSession().get();

		JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());

		coll.printSchema();
		if (this.partitions <= 0) {
			this.partitions = 1;
		}

		coll = coll.repartition(this.partitions).cache();

		int k = infos.size();

		Object[] feats = JavaConversions.mutableSeqAsJavaList(((Row) dataset.first()).getStruct(0).getAs("values"))
				.toArray();
		int dimensions = feats.length;

		System.out.println("n:" + all);
		System.out.println("k:" + k);
		System.out.println("partitions:" + this.partitions);
		System.out.println("dimensions:" + dimensions);

		CentroidAccumulator centAccu = new CentroidAccumulator(k, dimensions);

		JavaSparkContext.toSparkContext(sc).register(centAccu);

		double[] sizes = new double[infos.size()];

		for (Iterator<Row> iterator = infos.iterator(); iterator.hasNext();) {
			Row row = (Row) iterator.next();
			sizes[((Long) row.getAs("prediction")).intValue()] = ((Long) (row.getAs("count"))).doubleValue();
		}

		Broadcast<double[]> clustSizes = sc.broadcast(sizes);

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

		coll2.foreach((ForeachFunction<MyPair>) row ->{

			int cluster = Long.valueOf(row.getClust()).intValue();

			double[] slice = row.getFeats().toArray();

			double size = clustSizes.value()[cluster];

			for (int i = 0; i < slice.length; i++) {
				slice[i] = slice[i]/size;
			}

			centAccu.add(new DenseVector(slice), cluster);

		});


		Broadcast<DenseVector[]> centroids = sc.broadcast(centAccu.value());


		return coll2.map((MapFunction<MyPair, Double>) pivot ->{

			int clust = Long.valueOf(pivot.getClust()).intValue();
			double a = 0.0;
			double b = Double.MAX_VALUE;
			DenseVector[] list = centroids.value();
			for (int i = 0; i < list.length; i++) {
				if (i == clust) 
					a = ((Vectors.sqdist(pivot.getFeats(), list[i]))/(clustSizes.value()[i]-1.0));

				else
					b = Double.min(b, ((Vectors.sqdist(pivot.getFeats(), list[i])/clustSizes.value()[i])));
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
