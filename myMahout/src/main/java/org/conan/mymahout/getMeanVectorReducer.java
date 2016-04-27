package org.conan.mymahout;

import java.io.IOException;

import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.conf.Configuration;

import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.clustering.spectral.IntDoublePairWritable;

public class getMeanVectorReducer extends Reducer<NullWritable, IntDoublePairWritable, IntWritable, VectorWritable>{

   	@Override
    	public void reduce(NullWritable key, Iterable<IntDoublePairWritable> values, Context context)
        	throws IOException, InterruptedException{
		Configuration conf=context.getConfiguration();
		int rowNum = Integer.parseInt(conf.get("rowNum"));
		int colNum = Integer.parseInt(conf.get("colNum"));
		VectorWritable outputValue = new VectorWritable();
		double[] sum=new double[colNum];
      		for (IntDoublePairWritable value : values) {
        		int i = value.getKey();
        		sum[i] += value.getValue();
      		}
      		for (int i = 0; i < colNum; ++i) {
        		sum[i] /= rowNum;
      		}
      		outputValue.set(new DenseVector(sum));
      		context.write(new IntWritable(0), outputValue);
    	}
}
