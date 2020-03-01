package silhEval.research;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import silhEval.utils.Utils;

/**
 * Executable class that creates a dataset of random numbers as a 3D sphere with outliers in a d-dimensional space.
 * main parameters:
 * 	 elements- elements of the inner sphere
 * 	 outliers- elements on the surface of the outer sphere
 * 	 minorRay- ray of the inner sphere 
 * 	 mayorRay- ray of the outer sphere
 * 
 * @author Federico Altieri
 *
 */
public class FullRandomGen {

	public static void main(String[] args) throws IOException {
		
		int index = -1;
		
		int elements;
		int outliers;
		int dimensions;
		double majorRay;
		double minorRay;
		long genSeed;
		String file;
		
		
		if((index = Utils.index(args, "file")) != -1) {
			file = args[index+1];
		}
		else {
			throw new IllegalArgumentException("no target file");
		}
		
		if((index = Utils.index(args, "elements")) != -1) {
			elements = Integer.parseInt(args[index+1]);
		}
		else {
			throw new IllegalArgumentException("no number of lements");
		}
		
		if((index = Utils.index(args, "outliers")) != -1) {
			outliers = Integer.parseInt(args[index+1]);
		}
		else {
			outliers = 0;
		}
		
		if((index = Utils.index(args, "dimensions")) != -1) {
			dimensions = Integer.parseInt(args[index+1]);
		}
		else {
			dimensions = 1;
		}
		
		if((index = Utils.index(args, "minorRay")) != -1) {
			minorRay = Double.parseDouble(args[index+1]);
		}
		else {
			minorRay = 1.0;
		}
		
		if((index = Utils.index(args, "majorRay")) != -1) {
			majorRay = Double.parseDouble(args[index+1]);
		}
		else {
			majorRay = 1.0;
		}
				
		if((index = Utils.index(args, "genSeed")) != -1) {
			genSeed = Long.parseLong(args[index+1]);
		}
		else {
			genSeed = System.currentTimeMillis();
		}
		
		
		Random numbers = new Random();
		numbers.setSeed(genSeed);
		
		
		FileWriter writer = new FileWriter(file);
		double[] point = new double[dimensions];
		
		int ct = 0;
		while (ct < elements) {
			point = new double[dimensions];
			double sum = 0;
			for (int i = 0; i < point.length; i++) {
				point[i] = (numbers.nextDouble()*minorRay) - (minorRay/2);
				sum = sum + point[i]*point[i];
			}		
			
			if (Math.sqrt(sum) <= minorRay) {
				for (int i = 0; i < point.length-1; i++) {
					writer.write(point[i]+",");
				}
				writer.write(point[point.length-1]+"\n");
				ct++;
			}
		}
		
		ct = 0;
		while (ct < outliers) {
			double x = (numbers.nextDouble()*minorRay) - (minorRay/2);
			double y = (numbers.nextDouble()*minorRay) - (minorRay/2);
			double z = (numbers.nextDouble()*minorRay) - (minorRay/2);
			if (Math.sqrt((x*x)+(y*y)+(z*z)) <= minorRay) {
				double ratio = majorRay/(Math.sqrt((x*x)+(y*y)+(z*z)));
				x = ratio*x;
				y = ratio*y;
				z = ratio*z;
				writer.write(x+","+y+","+z+"\n");

				ct++;
			}
		}
		

		
		
		writer.close();
	}

}
