package org.conan.mymahout;

import java.io.IOException;

import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.conf.Configuration;

import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.clustering.spectral.IntDoublePairWritable;

public class getCovReducer extends Reducer<IntWritable, IntDoublePairWritable, IntWritable, VectorWritable>{
	private int colNum;
	private int rowNum;
    	@Override
    	protected void setup(Context context) throws IOException {
	      	Configuration conf = context.getConfiguration();
		colNum = Integer.parseInt(conf.get("colNum"));
		rowNum = Integer.parseInt(conf.get("rowNum"));
    	}
   	@Override
    	public void reduce(IntWritable key, Iterable<IntDoublePairWritable> values, Context context)
        	throws IOException, InterruptedException{
		VectorWritable outputValue = new VectorWritable();
		double[] sum=new double[colNum];
      
      		for (IntDoublePairWritable value : values) {
        		int i = value.getKey();
        		double val = value.getValue();        
        		sum[i] += val;
      		}
		for(int i=0; i<colNum; i++)sum[i]=sum[i]/(rowNum-1);
      		outputValue.set(new DenseVector(sum));
      		context.write(key, outputValue);
    	}
}
