package altierif.appxSilhEval;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import org.apache.spark.TaskContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.api.java.function.MapGroupsFunction;
import org.apache.spark.api.java.function.MapPartitionsFunction;
import org.apache.spark.api.java.function.ReduceFunction;
import org.apache.spark.ml.evaluation.Evaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.ParamPair;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import altierif.utils.Utils;
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

		double n = coll.count();

		List<Row> infos = coll.groupBy("prediction").count().collectAsList();
		
		coll.printSchema();
		if (this.partitions <= 0) {
			this.partitions = 1;
		}
		Column[] columnArray = new Column[1];
		columnArray[0] = new Column("index");
		
		coll = coll.repartition(this.partitions).cache();

		int k = infos.size();
		
		
		System.out.println("n:" + n);
		System.out.println("k:" + k);
		System.out.println("t:" + t);
		System.out.println("partitions:" + this.partitions);
		
		double[] clustSizes = new double[infos.size()];

		for (Iterator<Row> iterator = infos.iterator(); iterator.hasNext();) {
			Row row = (Row) iterator.next();
			clustSizes[((Long)row.getAs("prediction")).intValue()] = ((Long)(row.getAs("count"))).doubleValue();
		}
		
		if (this.t == 0) {
			t = (int) Math.ceil((1/(2*Math.pow(epsilon, 2)))*(Math.log(4)+Math.log(n)+Math.log(k)-Math.log(delta)));
		}

		double s0 = Math.ceil(Utils.log2(2*k/delta));


		/**PHASE 1**/

		/**
		 * MAP PHASE 1: extraction of the initial sample (through Poisson sampling)
		 * Creates copies of such sample, to be distributed to all buckets
		 **/

		return coll.mapPartitions((MapPartitionsFunction<Row, Row>) iter -> 
		{	
			List<Row> rows = new LinkedList<Row>();
			Row row;

			while (iter.hasNext()) {
				row = (Row) iter.next();

				double clustSize = clustSizes[((Long)row.getAs("prediction")).intValue()];
				
				double probab = (t/clustSize);

				if (clustSize > t) {

					if (Math.random() < probab)
					{
						for (int i = 0; i < this.partitions; i++) {
							rows.add(RowFactory.create(i , row.getAs("label"), row.getAs("features"), row.getAs("prediction"), probab, true));
						}
					}			
					rows.add(RowFactory.create(TaskContext.getPartitionId() , row.getAs("label"), row.getAs("features"), row.getAs("prediction"), probab, false));

				} else {
					
					for (int i = 0; i < this.partitions; i++) {
						rows.add(RowFactory.create(i , row.getAs("label"), row.getAs("features"), row.getAs("prediction"), 1.0, true));
					}

					rows.add(RowFactory.create(TaskContext.getPartitionId(), row.getAs("label"), row.getAs("features"), row.getAs("prediction"), 1.0, false));

				}
			}

			return rows.iterator();
		}
		, 	RowEncoder.apply(
				new StructType()
				.add("index", DataTypes.IntegerType)
				.add("label", DataTypes.DoubleType)
				.add("features", new StructType().add("type", DataTypes.LongType).add("values", DataTypes.createArrayType(DataTypes.DoubleType)))
				.add("prediction", DataTypes.LongType)
				.add("probab", DataTypes.DoubleType)
				.add("sample", DataTypes.BooleanType))
				)

		/**
		 * REDUCE PHASE 3: computes the pointwise silhouettes value with the sampled points and sums them
		 * 
		 * **/

		.repartition(this.partitions, columnArray)
		
		.mapPartitions((MapPartitionsFunction<Row, Double>) iter -> {

			@SuppressWarnings("unchecked")
			LinkedList<Vector>[] samples = new LinkedList[clustSizes.length];
			for (int i = 0; i < samples.length; i++) {
				samples[i] = new LinkedList<Vector>();
			}
			
			@SuppressWarnings("unchecked")
			LinkedList<Double>[] probabs = new LinkedList[clustSizes.length];
			for (int i = 0; i < probabs.length; i++) {
				probabs[i] = new LinkedList<Double>();
			}
			

			@SuppressWarnings("unchecked")
			LinkedList<Vector>[] rowsx0 = new LinkedList[clustSizes.length];
			for (int i = 0; i < rowsx0.length; i++) {
				rowsx0[i] = new LinkedList<Vector>();
			}
			while (iter.hasNext()) {
				Row pivot = (Row) iter.next();
				Object[] temp = JavaConversions.mutableSeqAsJavaList(pivot.getStruct(2).getAs("values")).toArray();
				double[] pivotarr = new double[temp.length];
				for (int r = 0; r < temp.length; r++) {
					pivotarr[r] = ((Number)temp[r]).doubleValue();
				}
				if (pivot.getBoolean(5)) {
					samples[Long.valueOf(pivot.getLong(3)).intValue()].add(Vectors.dense(pivotarr));
					probabs[Long.valueOf(pivot.getLong(3)).intValue()].add(pivot.getDouble(4));
				} else {
					rowsx0[Long.valueOf(pivot.getLong(3)).intValue()].add(Vectors.dense(pivotarr));
				}
			}

			/**POINTWISE SILHOUETTE SUMS**/
			double sum = 0.0;
			
			for (int i = 0; i < rowsx0.length; i++) {
				Iterator<Vector> iterx0 = rowsx0[i].iterator();
				
				while (iterx0.hasNext()) {
					Vector pivot = iterx0.next();
					double a = 0.0;
					double b = Double.MAX_VALUE;
					
					for (int j = 0; j < samples.length; j++) {
						Iterator<Vector> iterx1 = samples[j].iterator();
						Iterator<Double> iterprobabs = probabs[j].iterator();
						if (i == j) {
							while (iterx1.hasNext()) {
								Vector sample = iterx1.next();
								Double probab = iterprobabs.next();
								a = a + (Math.sqrt(Vectors.sqdist(pivot, sample)))/probab.doubleValue();
							}
							a = a/(clustSizes[i]-1.0);
						} else {
							double temp = 0.0;
							while (iterx1.hasNext()) {
								Vector sample = iterx1.next();
								Double probab = iterprobabs.next();
								temp = temp + (Math.sqrt(Vectors.sqdist(pivot, sample)))/probab.doubleValue();
							}
							b = Double.min(b, temp/clustSizes[i]);
						}
					}
					sum = sum + (b-a)/Double.max(a, b);
				}
			}
			ArrayList<Double> list = new ArrayList<>();
			list.add(sum);
			return list.iterator();
		},	Encoders.DOUBLE()
				)

		
		/**PHASE 4**/

		/**MAP-REDUCE PHASE 4: sums all the silhouette sums into an only one**/


//		return sums
				.reduce((ReduceFunction<Double>) (v1, v2 ) ->{
			return v1.doubleValue()+v2.doubleValue();
		})
				.doubleValue()/n;

	}

}
