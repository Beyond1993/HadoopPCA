package org.conan.mymahout;

import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Mapper;

import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.clustering.spectral.IntDoublePairWritable;

public class getCovMapper extends Mapper<IntWritable, VectorWritable, IntWritable, IntDoublePairWritable> {
	private Vector meanVector;
	private int colNum;
    	@Override
    	protected void setup(Context context) throws IOException {
	      	Configuration conf = context.getConfiguration();
		FileSystem fs = FileSystem.get(conf);
		colNum = Integer.parseInt(conf.get("colNum"));

		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(conf.get("meanVectorPath")), conf);
		IntWritable key = new IntWritable();
		VectorWritable value = new VectorWritable();
		reader.next(key, value);
		meanVector = value.get();
    	}
  	@Override
   	public void map(IntWritable key, VectorWritable value, Context context) throws IOException,
        InterruptedException{
	     	IntWritable outputKey = new IntWritable();
	     	IntDoublePairWritable outputValue = new IntDoublePairWritable();
		Vector rowVector = value.get();
	      	for (int i = 0; i < colNum; i++) {
			for (int j = 0; j <= i; j++) {
				double partialResult=(rowVector.get(i)-meanVector.get(i)) * (rowVector.get(j)-meanVector.get(j));
			  	outputKey.set(i);
			  	outputValue.setKey(j);
			  	outputValue.setValue(partialResult);
			  	context.write(outputKey, outputValue);
				if(i==j) continue;
			  	outputKey.set(j);
			  	outputValue.setKey(i);
			  	outputValue.setValue(partialResult);
			  	context.write(outputKey, outputValue);
			}
	      	}

      }
}
