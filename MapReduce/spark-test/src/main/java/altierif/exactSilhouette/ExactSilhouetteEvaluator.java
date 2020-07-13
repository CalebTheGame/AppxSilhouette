package altierif.exactSilhouette;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.api.java.function.MapPartitionsFunction;
import org.apache.spark.api.java.function.ReduceFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.evaluation.Evaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.ParamPair;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import altierif.appxSilhEval.PPSSilhouetteEstimtor;
import altierif.utils.MyPair;
import altierif.utils.SamplePair;
import scala.collection.JavaConversions;

/**
 * @author Federico Altieri
 * 
 * Class that implements the Spark implementation of the Map-Reduce algorithm for exact silhouette computation
 * with euclidean distance
 *
 */
@SuppressWarnings("serial")
public class ExactSilhouetteEvaluator extends Evaluator {

	private String uid;
	private int partitions;
	private long bufsize;

	public ExactSilhouetteEvaluator() {
		this("ExactShilhEval" + Long.toString(System.currentTimeMillis()));
	}

	/**
	 * 
	 * @param uid objects's UID
	 */
	public ExactSilhouetteEvaluator(String uid) {
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

	public void setBufsize(int bufsize) {
		this.bufsize = bufsize;
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
		System.out.println("partitions:" + this.partitions);

		Broadcast<int[]> dataInfo = sc.broadcast(new int[] { k });

		double[] clustersSizes = new double[infos.size()];

		for (Iterator<Row> iterator = infos.iterator(); iterator.hasNext();) {
			Row row = (Row) iterator.next();
			clustersSizes[((Long)row.getAs("prediction")).intValue()] = ((Long)(row.getAs("count"))).doubleValue();
		}
		
		Broadcast<double[]> clustSizes = sc.broadcast(clustersSizes);

		coll.repartition(this.partitions).cache();		
		
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
		
		Broadcast<List<MyPair>> link = sc.broadcast(coll2.collectAsList());
		
		return coll2.map((MapFunction<MyPair, Double>) pivot ->{

			int clust = Long.valueOf(pivot.getClust()).intValue();
			double a = 0.0;
			double b = Double.MAX_VALUE;
			double[] distsums = new double[dataInfo.value()[0]];
			Iterator<MyPair> iter = link.value().iterator();
			while (iter.hasNext()) {
				MyPair s = (MyPair) iter.next();
				distsums[Long.valueOf(s.getClust()).intValue()] = distsums[Long.valueOf(s.getClust()).intValue()] + (Math.sqrt(Vectors.sqdist(pivot.getFeats(), s.getFeats())));
			}
			
			for (int i = 0; i < distsums.length; i++) {
				if (i == clust) {
					a = distsums[i]/(clustSizes.value()[i]-1.0);
				}
				else{
					b = Double.min(b, (distsums[i]/clustSizes.value()[i]));
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
