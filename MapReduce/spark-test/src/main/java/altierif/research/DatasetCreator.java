package altierif.research;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.util.DatasetUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import altierif.utils.Utils;

public class DatasetCreator {
	
	@Deprecated
	public static void generateAndStore(String[] args, SparkSession spark) throws IOException {

		String file = Utils.findStringParam("path", args);
		double n = Utils.findDoubleParam("n", args);
		long genSeed = Utils.findLongParam("genSeed", args);
		int dim = Utils.findIntParam("dimensions", args);
		int k = Utils.findIntParam("k", args);
		
		
		Random numbers = new Random();
		numbers.setSeed(genSeed);
		List<Row> list=new ArrayList<Row>();
		
		for (double i = 0; i < n; i++) {
			double[] elements = new double[dim];
			for (int j = 0; j < elements.length; j++) {
				elements[j] = numbers.nextDouble()*1000000;
			}
			list.add(RowFactory.create(i, new DenseVector(elements)));
		}		
		
		List<org.apache.spark.sql.types.StructField> listOfStructField = new ArrayList<org.apache.spark.sql.types.StructField>();
		
		listOfStructField.add(DataTypes.createStructField("label", DataTypes.DoubleType, true));
		listOfStructField.add(DataTypes.createStructField("features", org.apache.spark.ml.linalg.SQLDataTypes.VectorType(), true));

		StructType structType = DataTypes.createStructType(listOfStructField);

		Dataset<Row> data = spark.createDataFrame(list,structType)
				.repartition()
				.cache();

		// Trains a k-means model.
		KMeans kmeans = new KMeans().setK(k).setSeed(1L);
		KMeansModel model = kmeans.fit(data);
		// Make predictions
		data = model.transform(data);
		
		data.write()
			.format("json")
			.option("header", true)
			.option("codec", "org.apache.hadoop.io.compress.GzipCodec")
			.save(file);
		
	}

	public static void importAndStore(String[] args, SparkSession spark) throws FileNotFoundException {
		
		String dataFile = Utils.findStringParam("source", args);
		String file = Utils.findStringParam("path", args);
		

		System.out.println("inizio parsing");

		System.out.println("creazione e stampa dataset");
//		Dataset<Row> data = spark.createDataFrame(list,structType);
		spark.read().format("csv")
			      .option("header", "true")
	//		      .schema(schema)
			      .csv(dataFile)
			      .withColumn("label", org.apache.spark.sql.functions.monotonically_increasing_id())
			      .repartition(Integer.parseInt(spark.conf().get("spark.executor.instances")))
			      .map((MapFunction<Row, Row>) row -> 
			      {
			    	  
			    	  double[] elements = new double[row.length()-2];
						int[] indexes = new int[row.length()-2];
						for (int i = 0; i < row.length()-2; i++) {
							indexes[i] = i;
							elements[i] = Double.parseDouble(row.getString(i));
						}
						return RowFactory.create(Long.valueOf(row.getLong(row.length()-1)).doubleValue(),new DenseVector(elements),Long.parseLong(row.getString(row.length()-2)));
			      },
			      RowEncoder.apply(
							new StructType()
							.add(DataTypes.createStructField("label", DataTypes.DoubleType, true))
							.add(DataTypes.createStructField("features", org.apache.spark.ml.linalg.SQLDataTypes.VectorType(), true))
							//				.add("features", org.apache.spark.ml.linalg.SQLDataTypes.VectorType())
							.add(DataTypes.createStructField("prediction", DataTypes.LongType, true))
							) 
			    		  )
////		System.out.println("dataset creato");
////		data.repartition(Integer.parseInt(spark.conf().get("spark.executor.instances")));
////		data.cache();
////		System.out.println("inizio scrittura");
////		data
		.write()
		
			.format("json")
			.option("header", true)
			.option("codec", "org.apache.hadoop.io.compress.GzipCodec")
			.save(file);
	}
	
public static void importAndStore2(String[] args, SparkSession spark) throws FileNotFoundException {
		
		String dataFile = Utils.findStringParam("source", args);
		String file = Utils.findStringParam("path", args);
		

		System.out.println("inizio parsing");

		System.out.println("creazione e stampa dataset");
		Dataset<Row> data = 
		spark.read().format("csv")
			      .option("header", "true")
	//		      .schema(schema)
			      .csv(dataFile)
			      .withColumn("label", org.apache.spark.sql.functions.monotonically_increasing_id())
			      .repartition(Integer.parseInt(spark.conf().get("spark.executor.instances")))
			      .map((MapFunction<Row, Row>) row -> 
			      {		    	  
			    	  double[] elements = new double[row.length()-2];
						short[] indexes = new short[row.length()-2];
						for (int i = 0; i < row.length()-2; i++) {
							indexes[i] = Integer.valueOf(i).shortValue();
							elements[i] = Double.parseDouble(row.getString(i));
						}
						return RowFactory.create(Long.valueOf(row.getLong(row.length()-1)).doubleValue(),elements,Long.parseLong(row.getString(row.length()-2)));
			      },
			      RowEncoder.apply(
							new StructType()
							.add(DataTypes.createStructField("label", DataTypes.DoubleType, false))
							.add(DataTypes.createStructField("features", ArrayType.apply(DataTypes.DoubleType, false), false))
							//.add(DataTypes.createStructField("features", new VectorUDT(), false))
							//.add(DataTypes.createStructField("features", org.apache.spark.ml.linalg.SQLDataTypes.VectorType(), true))
							//
							.add(DataTypes.createStructField("prediction", DataTypes.LongType, false))
							) 
			    		  );
		
//		VectorAssembler ass = new VectorAssembler();
//		String[] arr = {"featurest"};
//		ass.setInputCols(arr);
//		ass.setOutputCol("features");
//		DatasetUtils.columnToVector(data, "features");
		data.printSchema();
		data.write()
		
			.format("json")
			.option("header", true)
			.option("codec", "org.apache.hadoop.io.compress.GzipCodec")
			.save(file);
	}

}
