package altierif.utils;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;

/**
 * Collector of various utilities that implements various functions.
 * 
 * @author Federico Altieri
 *
 */
public class Utils {
	
	/** 
	 * Gets CPU time in nanoseconds. 
	 * */
	public static long getCpuTime( ) {
	    ThreadMXBean bean = ManagementFactory.getThreadMXBean( );
	    return bean.isCurrentThreadCpuTimeSupported( ) ?
	        bean.getCurrentThreadCpuTime( ) : 0L;
	}
	 
	/** 
	 * Gets user time in nanoseconds. 
	 * */
	public static long getUserTime( ) {
	    ThreadMXBean bean = ManagementFactory.getThreadMXBean( );
	    return bean.isCurrentThreadCpuTimeSupported( ) ?
	        bean.getCurrentThreadUserTime( ) : 0L;
	}

	/** 
	 * Get system time in nanoseconds. 
	 * */
	public static long getSystemTime( ) {
	    ThreadMXBean bean = ManagementFactory.getThreadMXBean( );
	    return bean.isCurrentThreadCpuTimeSupported( ) ?
	        (bean.getCurrentThreadCpuTime( ) - bean.getCurrentThreadUserTime( )) : 0L;
	}
	
	/**
	 * Finds the index of the first occurrence of a {@link String} within an array of {@link String} instances. 
	 * 
	 * @param tokens the list of tokens as array of {@link String}
	 * @param token name of the token
	 * @return the index of the found token. -1 if not found
	 */
	public static int index(String[] tokens, String token) {
		for(int i=0; i<tokens.length; i++) {
			if(tokens[i].equals(token)) {
				return i;
			}
		}
		return -1;
	}
	
	/**
	 * Returns the base 2 logarithm of integer a. 
	 * 
	 * @param d The value to compute the base 2 logarithm.
	 * @return The value of the logarithm
	 */
	public static double log2(double d) {
		return Math.log(d)/Math.log(2);
	}
	
	/**
	 * Parses a parameter of type double from a collection of parameters expressed as an array of {@link String} 
	 * structured as "parname1, value1, parname2, value2,...".
	 * 
	 * @param parName name of the parameter
	 * @param args the list of parameters as array of {@link String}
	 * @return the parameter value as a double
	 * @throws IllegalArgumentException if something goes wrong (not found of incompatible types)
	 */
	public static double findDoubleParam(String parName, String[] args) throws IllegalArgumentException {
		int index;
		if ((index  = Utils.index(args, parName)) != -1) {
			try {
				return Double.parseDouble(args[index + 1]);
			} catch (Exception e) {
				throw new IllegalArgumentException("Illegal value for parameter "+parName+".");
			}
		} else {
			throw new IllegalArgumentException("Parameter "+parName+" not found.");
		}
	}
	
	/**
	 * Parses a parameter of type {@link String} from a collection of parameters expressed as an array of {@link String} 
	 * structured as "parname1, value1, parname2, value2,...".
	 * 
	 * @param parName name of the parameter
	 * @param args the list of parameters as array of {@link String}
	 * @return the parameter value as a {@link String}
	 * @throws IllegalArgumentException if something goes wrong (not found of incompatible types)
	 */
	public static String findStringParam(String parName, String[] args) throws IllegalArgumentException {
		int index;
		if ((index  = Utils.index(args, parName)) != -1) {
			try {
				return (args[index + 1]);
			} catch (Exception e) {
				throw new IllegalArgumentException("Illegal value for parameter "+parName+".");
			}
		} else {
			throw new IllegalArgumentException("Parameter "+parName+" not found.");
		}
	}

	public static long findLongParam(String parName, String[] args) throws IllegalArgumentException {
		int index;
		if ((index  = Utils.index(args, parName)) != -1) {
			try {
				return Long.parseLong(args[index + 1]);
			} catch (Exception e) {
				throw new IllegalArgumentException("Illegal value for parameter "+parName+".");
			}
		} else {
			throw new IllegalArgumentException("Parameter "+parName+" not found.");
		}
	}

	public static int findIntParam(String parName, String[] args) throws IllegalArgumentException {
		int index;
		if ((index  = Utils.index(args, parName)) != -1) {
			try {
				return Integer.parseInt(args[index + 1]);
			} catch (Exception e) {
				throw new IllegalArgumentException("Illegal value for parameter "+parName+".");
			}
		} else {
			throw new IllegalArgumentException("Parameter "+parName+" not found.");
		}
	}

	public static boolean findBooleanParam(String parName, String[] args) throws IllegalArgumentException {
		int index;
		if ((index  = Utils.index(args, parName)) != -1) {
			try {
				return Boolean.parseBoolean(args[index + 1]);
			} catch (Exception e) {
				throw new IllegalArgumentException("Illegal value for parameter "+parName+".");
			}
		} else {
			throw new IllegalArgumentException("Parameter "+parName+" not found.");
		}
	}

}
