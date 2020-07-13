package altierif.appxSilhEval;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import org.apache.spark.TaskContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
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
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import altierif.appxSilhEval.PPSSilhouetteEstimtor;
import altierif.utils.CentroidAccumulator;
import altierif.utils.MyPair;
import scala.collection.JavaConversions;

/**
 * @author Federico Altieri
 * 
 * Class that implements the Spark implementation of the Map-Reduce algorithm for silhouette computation
 * with euclidean distance, according to definition of Frahler and Solher
 *
 */
@SuppressWarnings("serial")
public class FrahlingSolherSilhouette extends Evaluator {

	private String uid;
	private int partitions;
	private long bufsize;
	public double failures;

	public FrahlingSolherSilhouette() {
		this("ExactShilhEval" + Long.toString(System.currentTimeMillis()));
	}

	/**
	 * @param uid objects's UID
	 */
	public FrahlingSolherSilhouette(String uid) {
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
		
		Broadcast<int[]> dataInfo = sc.broadcast(new int[] { k });

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
		
		List<MyPair>[] toBroad = new List[k];
		List<Long> selected;
		
//		for (int i = 0; i < k; i++) {
//			selected = new ArrayList<Long>();
//			selected.add(new Long(i));
//			toBroad[i] = coll2.filter(coll.col("prediction").isInCollection(selected)).collectAsList();
//		}
		
		Iterator<Row> itera = infos.iterator();

		@SuppressWarnings("unchecked")
		List<Row>[] linkst = new List[k];
		for (int j = 0; j < linkst.length; j++) {
			selected = new ArrayList<Long>();
			selected.add((Long) itera.next().getAs(0));
			toBroad[j] = coll.filter(coll.col("prediction").isInCollection(selected))
					.map((MapFunction<Row, MyPair>) row -> 
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
					)
					.collectAsList();	
		}
		
		Broadcast<List<MyPair>[]> links = sc.broadcast(toBroad);
		
		return coll2.map((MapFunction<MyPair, Double>) pivot ->{

			int clust = Long.valueOf(pivot.getClust()).intValue();
			int mincl = 0;
			double mindist = Double.MAX_VALUE;
			for (int i = 0; i < dataInfo.value()[0]; i++) {
				if (i != clust)
					if ((Math.sqrt(Vectors.sqdist(pivot.getFeats(), centroids.value()[i]))) < mindist ) {
						mindist = (Math.sqrt(Vectors.sqdist(pivot.getFeats(), centroids.value()[i])));
						mincl = i;
					}
			}
			
			
			double a = 0.0;
			double b = 0.0;
			Iterator<MyPair> iter = links.value()[clust].iterator();
			while (iter.hasNext()) {
				MyPair s = (MyPair) iter.next();
				a = a + (Math.sqrt(Vectors.sqdist(pivot.getFeats(), s.getFeats())));
			}
			a = a/(clustSizes.value()[clust]-1.0);
			
			iter = links.value()[mincl].iterator();
			while (iter.hasNext()) {
				MyPair s = (MyPair) iter.next();
				b = b + (Math.sqrt(Vectors.sqdist(pivot.getFeats(), s.getFeats())));
			}
			b = b/(clustSizes.value()[clust]);
			
			return Double.valueOf((b-a)/Double.max(a, b));
		},
				Encoders.DOUBLE())
		
		/** sums all the silhouette sums into an only one**/		
				
		.reduce((ReduceFunction<Double>) (s1,s2) ->{
			return (s1.doubleValue() + s2.doubleValue());
		}).doubleValue()/all;

	}
}


