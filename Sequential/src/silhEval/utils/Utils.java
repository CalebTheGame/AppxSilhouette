package silhEval.utils;

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
	 * @param tokens
	 * @param token
	 * @return
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

}
