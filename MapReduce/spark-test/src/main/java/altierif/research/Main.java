package altierif.research;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.evaluation.SquaredEuclideanSilhouette;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.util.SizeEstimator;

import altierif.appxSilhEval.PPSSilhouetteEstimtor;
import altierif.appxSilhEval.UniformSamplingClusteringEvaluator;
import altierif.exactSilhouette.ExactClusteringEvaluator;
import altierif.utils.Utils;
import altierif.utils.Logger;

public class Main {

	/**
	 * main for tests
	 * 
	 * actual main perform parsing of dataset from file and computation of both
	 * silhouettes values
	 * 
	 * @param args path - relative path to file delta epsilon
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		// Create a SparkSession.
		SparkSession spark = SparkSession.builder().appName("Silhouette appx test").getOrCreate();

		spark.sparkContext().setLogLevel("ERROR");

		String mode = "";

		mode = Utils.findStringParam("mode", args);

		if (mode.equals("generation")) {
			DatasetCreator.generateAndStore(args, spark);
			System.out.println("Done!");
			return;
		}

		if (mode.equals("importDb")) {
			DatasetCreator.importAndStore(args, spark);
			System.out.println("Done!");
			return;
		}
		
		if (mode.equals("importDb2")) {
			DatasetCreator.importAndStore2(args, spark);
			System.out.println("Done!");
			return;
		}

		if ((Utils.index(args, "performance")) != -1) {
			performanceTest(args, spark);
			System.out.println("Done!");
			return;
		}

		if ((Utils.index(args, "performanceSpark")) != -1) {
			performanceSparkTest(args, spark);
			System.out.println("Done!");
			return;
		}

		if ((Utils.index(args, "exactOnly")) != -1) {
			exactSilhOnly(args, spark);
			System.out.println("Done!");
			return;
		}

		if ((Utils.index(args, "naive")) != -1) {
			naiveSampling(args, spark);
			System.out.println("Done!");
			return;
		}

		spark.close();

	}

	private static void exactSilhOnly(String[] args, SparkSession spark) {
		// Loads data.
		String outfile = Utils.findStringParam("outfile", args);
		boolean append = Utils.findBooleanParam("append", args);
		Logger csvStats = new Logger(outfile, append);

		Dataset<Row> loaded = getDataset(Utils.findStringParam("path", args), spark);

		ExactClusteringEvaluator naiveEval = new ExactClusteringEvaluator();
		long startTimeNaive = System.nanoTime();
		double silhouette = naiveEval.evaluate(loaded);
		long endTimeNaive = System.nanoTime();
		double overallTime = (endTimeNaive - startTimeNaive) / 10e8;
		System.out.println("Silhouette with squared euclidean distance = " + silhouette);
		csvStats.writeLn("Exact," + loaded.count() + "," + loaded.groupBy("prediction").count().collectAsList().size()
				+ "," + silhouette + "," + overallTime);
	}

	private static void performanceSparkTest(String[] args, SparkSession spark) {

		Dataset<Row> loaded;

		loaded = getDataset(Utils.findStringParam("path", args), spark).repartition().cache();

		String outfile = Utils.findStringParam("outfile", args);
		boolean append = Utils.findBooleanParam("append", args);
		Logger csvStats = new Logger(outfile, append);

		if (!append) {
			csvStats.writeLn("type,n,k,silh,time");
		}
		
		loaded = loaded.map((MapFunction<Row, Row>) row -> 
	      {		    	  
				return row;
	      },
	      RowEncoder.apply(
					new StructType()
					.add(DataTypes.createStructField("features", ArrayType.apply(DataTypes.DoubleType, false), false))
					//.add(DataTypes.createStructField("features", new VectorUDT(), false))
					//.add(DataTypes.createStructField("features", org.apache.spark.ml.linalg.SQLDataTypes.VectorType(), true))
					//
					.add(DataTypes.createStructField("label", DataTypes.DoubleType, false))
					.add(DataTypes.createStructField("prediction", DataTypes.LongType, false))
					) 
	    		  );

		ClusteringEvaluator silh = new ClusteringEvaluator();
		
		loaded.printSchema();

		long startTimeAppx = System.nanoTime();
		double appxSilhouette = silh.evaluate(loaded);
		long endTimeAppx = System.nanoTime();
		double overallTime = (endTimeAppx - startTimeAppx) / 10e8;

		csvStats.writeLn("SQEQ," + loaded.count() + "," + loaded.groupBy("prediction").count().collectAsList().size()
				+ "," + appxSilhouette + "," + overallTime);

		spark.stop();

	}

	private static void performanceTest(String[] args, SparkSession spark) {

		// ...
		Dataset<Row> loaded;

		if (Utils.findBooleanParam("parsed", args)) {
			loaded = getDataset(Utils.findStringParam("path", args), spark);
//				.repartition(Integer.parseInt(spark.conf().get("spark.executor.instances")))
//				.cache();
		} else {

			loaded = spark.read().format("csv").option("header", "true").option("inferSchema", "true")
					.csv(Utils.findStringParam("path", args))
					.withColumn("label", org.apache.spark.sql.functions.monotonically_increasing_id()).repartition()
					.map((MapFunction<Row, Row>) row -> {

						double[] elements = new double[row.length() - 2];
						int[] indexes = new int[row.length() - 2];
						for (int i = 0; i < row.length() - 2; i++) {
							indexes[i] = i;
							elements[i] = Double.parseDouble(row.getString(i));

						}

						return RowFactory.create(
								new GenericRowWithSchema(new Object[] { 1L, new DenseVector(elements) },
										new StructType().add("type", DataTypes.LongType).add("values",
												DataTypes.createArrayType(DataTypes.DoubleType))),
								Long.valueOf(row.getLong(row.length() - 1)).doubleValue(),
								Long.parseLong(row.getString(row.length() - 2)));
					}, RowEncoder
							.apply(new StructType().add("features", new StructType().add("type", DataTypes.LongType)
									.add("values", DataTypes.createArrayType(DataTypes.DoubleType)))
//						.add(DataTypes.createStructField("features", org.apache.spark.ml.linalg.SQLDataTypes.VectorType(), true))
									.add(DataTypes.createStructField("label", DataTypes.DoubleType, true))
//						.add(DataTypes.createStructField("features", org.apache.spark.ml.linalg.SQLDataTypes.VectorType(), true))
									.add(DataTypes.createStructField("prediction", DataTypes.LongType, true))));
			loaded.printSchema();
			System.out.println(loaded.first());
//			String[] toAssemble = new String[loaded.first().length()-2];
//			for (int i = 0; i < toAssemble.length; i++) {
//				toAssemble[i] = ("element"+i);
//			}
//			VectorAssembler builder = new VectorAssembler().setInputCols(toAssemble).setOutputCol("features");
//			loaded = builder.transform(loaded);
//			for (int i = 0; i < toAssemble.length; i++) {
//				loaded = loaded.drop(toAssemble[i]);
//			}
//			loaded = loaded.withColumnRenamed("cluster", "prediction");

		}

		String outfile = Utils.findStringParam("outfile", args);
		boolean append = Utils.findBooleanParam("append", args);
		Logger csvStats = new Logger(outfile, append);

		if (!append) {
			csvStats.writeLn("type,n,k,silh,time");
		}

		double delta;
		double epsilon;
		int t;
		int parts = 0;
		try {
			delta = Utils.findDoubleParam("delta", args);
			epsilon = Utils.findDoubleParam("epsilon", args);
			parts = Utils.findIntParam("partitions", args);

		} catch (IllegalArgumentException e) {
			delta = PPSSilhouetteEstimtor.DEFAULT_DELTA;
			epsilon = PPSSilhouetteEstimtor.DEFAULT_EPS;
		}

		try {
			t = Utils.findIntParam("t", args);

		} catch (IllegalArgumentException e) {
			t = 0;
		}

//ApproxClusteringEvaluator appxEval = new ApproxClusteringEvaluator(delta, epsilon);

		PPSSilhouetteEstimtor appxEval = new PPSSilhouetteEstimtor(
				"AppxShilhEval" + Long.toString(System.currentTimeMillis()), delta, epsilon, t);

		if (parts != 0) {
			appxEval.setPartitions(parts);
		} else {
			double totsizeBytes = Long.valueOf(SizeEstimator.estimate(loaded.first())).doubleValue() * (loaded.count());
			System.out.println(totsizeBytes);
			double fract = (spark.sparkContext().conf().getSizeAsBytes("spark.executor.memory") *
			// Double.parseDouble(spark.sparkContext().conf().get("spark.memory.fraction"))
			// *
					0.2 *
					// Double.parseDouble(spark.sparkContext().conf().get("spark.shuffle.safetyFraction"))
					0.6 * 0.4 // algo safe zone
			) / (Integer.parseInt(spark.conf().get("spark.executor.cores")) * 1.0);
			System.out.println(fract);
			appxEval.setPartitions((int) (Math.ceil(Math.ceil(totsizeBytes / fract)
					/ (Integer.parseInt(spark.conf().get("spark.executor.instances")) * 1.0)))
					* Integer.parseInt(spark.conf().get("spark.executor.instances")));
		}

		long startTimeAppx = System.nanoTime();
		double appxSilhouette = appxEval.evaluate(loaded);
		long endTimeAppx = System.nanoTime();
		double overallTime = (endTimeAppx - startTimeAppx) / 10e8;

		csvStats.writeLn("PPS," + loaded.count() + "," + loaded.groupBy("prediction").count().collectAsList().size()
				+ "," + appxSilhouette + "," + overallTime);

		spark.stop();

	}

	private static void naiveSampling(String[] args, SparkSession spark) {

		// ...

		Dataset<Row> loaded = getDataset(Utils.findStringParam("path", args), spark);

		String outfile = Utils.findStringParam("outfile", args);
		boolean append = Utils.findBooleanParam("append", args);
		Logger csvStats = new Logger(outfile, append);

		if (!append) {
			csvStats.writeLn("type,n,k,silh,time");
		}

		double delta;
		double epsilon;
		try {
			delta = Utils.findDoubleParam("delta", args);
			epsilon = Utils.findDoubleParam("epsilon", args);

		} catch (IllegalArgumentException e) {
			delta = UniformSamplingClusteringEvaluator.DEFAULT_DELTA;
			epsilon = UniformSamplingClusteringEvaluator.DEFAULT_EPS;
		}

		UniformSamplingClusteringEvaluator appxEval = new UniformSamplingClusteringEvaluator(delta, epsilon);
		long startTimeAppx = System.nanoTime();
		double appxSilhouette = appxEval.evaluate(loaded);
		long endTimeAppx = System.nanoTime();
		double overallTime = (endTimeAppx - startTimeAppx) / 10e8;

		csvStats.writeLn("Naive," + loaded.count() + "," + loaded.groupBy("prediction").count().collectAsList().size()
				+ "," + appxSilhouette + "," + overallTime);

		spark.stop();

	}

	private static Dataset<Row> getDataset(String path, SparkSession spark) {

		return new DataFrameReader(spark).format("json").option("header", true)
				.option("codec", "org.apache.hadoop.io.compress.GzipCodec").load(path);
	}

}
