package altierif.utils;

import java.io.FileWriter;
import java.io.IOException;

/**
 * Class that handles a Log file with algorithm output
 * 
 * @author Federico Altieri
 *
 */
public class Logger {
	/**
	 * {@link FileWriter} instance that handles the file
	 */
	private FileWriter writer;

	/**
	 * Initializes the logger with the file whose path is passed as parameter. Writes an error message if it fails.
	 * 
	 * @param file The path to file.
	 */
	public Logger(String file, boolean append) {
		super();
		try {
			this.writer = new FileWriter(file, append);
		} catch (IOException e) {
			System.err.println("Failed to open filewriter with file "+file );
			e.printStackTrace();
		}
	}
	
	/**
	 * Appends a {@link String} to the current file. Writes an error message if it fails.
	 * 
	 * @param toWrite The {@link String} to append.
	 */
	public void write(String toWrite) {
		try {
			writer.write(toWrite);
			writer.flush();
		} catch (IOException e) {
			System.err.println("Failed to write log");
			e.printStackTrace();
		}		
	}
	
	/**
	 * Closes the file. Returns false if it fails.
	 * 
	 * @return true if success. False Elsewhere
	 */
	public boolean close() {
		try {
			writer.flush();
			writer.close();
		} catch (IOException e) {
			return false;
		}
		return true;
	}
	
	/**
	 * @param toWrite
	 */
	public void writeLn(String toWrite) {
		try {
			writer.write(toWrite+"\n");
			writer.flush();
		} catch (IOException e) {
			System.err.println("Failed to write log");
			e.printStackTrace();
		}		
	}
	
	
}
