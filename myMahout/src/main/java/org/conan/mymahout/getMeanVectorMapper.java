package org.conan.mymahout;

import java.io.IOException;

import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.IntWritable;

import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.clustering.spectral.IntDoublePairWritable;

public class getMeanVectorMapper extends Mapper<IntWritable, VectorWritable, NullWritable, IntDoublePairWritable> {

  	@Override
   	public void map(IntWritable key, VectorWritable value, Context context) throws IOException,
        InterruptedException{
		IntDoublePairWritable outputValue = new IntDoublePairWritable();
      		Vector vector = value.get();    
      		for (Vector.Element e : vector.nonZeroes()) {
        	outputValue.setKey(e.index());
        	outputValue.setValue(e.get());
        	context.write(NullWritable.get(), outputValue);
      }
    }
}
