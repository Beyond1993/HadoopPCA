package org.conan.mymahout;

import java.util.ArrayList;
import java.net.URI;

import java.io.PrintWriter;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;

import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.clustering.spectral.IntDoublePairWritable;
import org.apache.mahout.math.SingularValueDecomposition;

public class Driver {
	private static int rowNum,colNum;
	public static void main(String args[]) throws Exception {
		String input = args[0];
		String output = args[1];
		String tmpdir = args[2];
		// command line argument defined as it
		ArrayList<Integer> list=GetMahoutVectorFromTXT.readTXTWriteMahoutVector(input,tmpdir+"/sequentilizedMatrix/seq.mat");
		rowNum=list.get(0);
		colNum=list.get(1);
		getMeanVector(tmpdir+"/sequentilizedMatrix/seq.mat", tmpdir + "/meanVector");
		getCov(tmpdir+"/sequentilizedMatrix/seq.mat", tmpdir+"/cov",tmpdir + "/meanVector/part-r-00000");
		getEigenVectorMatrix(tmpdir + "/cov/part-r-00000",output,tmpdir+"/sequentilizedMatrix/seq.mat",tmpdir + "/meanVector/part-r-00000");
	}

	private static void getMeanVector(String input, String output)
			throws Exception {
		Configuration conf=new Configuration();
		conf.set("rowNum",String.valueOf(rowNum));
		conf.set("colNum",String.valueOf(colNum));
		Job job = new Job(conf);
		job.setJarByClass(Driver.class);
		//JobConf job = new JobConf(conf,Driver.class);
		job.setJobName("get mean vector");

		job.setMapperClass(getMeanVectorMapper.class);
		job.setReducerClass(getMeanVectorReducer.class);

		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setMapOutputKeyClass(NullWritable.class);
		job.setMapOutputValueClass(IntDoublePairWritable.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(VectorWritable.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);


		FileSystem fs = FileSystem.get(URI.create(output), conf);
		fs.delete(new Path(output), true);

		SequenceFileInputFormat.addInputPath(job, new Path(input));
		SequenceFileOutputFormat.setOutputPath(job, new Path(output));	

		SequenceFileInputFormat.setMinInputSplitSize(job,500000);	
		SequenceFileInputFormat.setMaxInputSplitSize(job,750000);	
//		job.setNumMapTasks(15);
		job.setNumReduceTasks(1);
	
		long start = System.currentTimeMillis();
		job.waitForCompletion(true);
		long end = System.currentTimeMillis();
		System.out.println(String.format("Runtime for the mean Job is %d ms",end - start));
	}
	private static void getCov(String input, String output, String meanVectorPath)
			throws Exception {
		Configuration conf=new Configuration();
		conf.set("meanVectorPath",meanVectorPath);
		conf.set("rowNum",String.valueOf(rowNum));
		conf.set("colNum",String.valueOf(colNum));
		Job job = new Job(conf);
		job.setJarByClass(Driver.class);
		//JobConf job = new JobConf(conf,Driver.class);
		job.setJobName("get cov Matrix");

		job.setMapperClass(getCovMapper.class);
		job.setReducerClass(getCovReducer.class);

		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(IntDoublePairWritable.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(VectorWritable.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);

		FileSystem fs = FileSystem.get(URI.create(output), conf);
		fs.delete(new Path(output), true);

		SequenceFileInputFormat.addInputPath(job, new Path(input));
		SequenceFileOutputFormat.setOutputPath(job, new Path(output));
		SequenceFileInputFormat.setMinInputSplitSize(job,500000);	
		SequenceFileInputFormat.setMaxInputSplitSize(job,750000);	
	//	job.setNumMapTasks(15);
		job.setNumReduceTasks(1);
	
		long start = System.currentTimeMillis();
		job.waitForCompletion(true);
		long end = System.currentTimeMillis();
		System.out.println(String.format("Runtime for the cov Job is %d ms",end - start));
	}
	private static void getEigenVectorMatrix(String input, String output, String originMatrix,String meanVectorPath)
			throws Exception {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(input), conf);

		IntWritable key = new IntWritable();
		VectorWritable value = new VectorWritable();
		Matrix cov=new DenseMatrix(colNum,colNum);

		while(reader.next(key, value)){
			int rowIdx=key.get();
			Vector rowVector=value.get();
			cov.assignRow(rowIdx,rowVector);
		}
		

		SingularValueDecomposition svd=new SingularValueDecomposition(cov);
		
//		long start2 = System.currentTimeMillis();
		Matrix emat=svd.getV();
//		long end2 = System.currentTimeMillis();
//		System.out.println(String.format("Runtime for svd is %d ms",end2 - start2));

	      	Configuration conf3 = new Configuration();
		FileSystem fs3 = FileSystem.get(conf3);

		SequenceFile.Reader reader3 = new SequenceFile.Reader(fs3, new Path(meanVectorPath), conf3);
		IntWritable key3 = new IntWritable();
		VectorWritable value3 = new VectorWritable();
		reader3.next(key3, value3);
		Vector meanVector = value3.get();

		Configuration conf1 = new Configuration();
		FileSystem fs1 = FileSystem.get(conf1);
		SequenceFile.Reader reader1 = new SequenceFile.Reader(fs1, new Path(originMatrix), conf1);

		IntWritable key1 = new IntWritable();
		VectorWritable value1 = new VectorWritable();
		Matrix origin=new DenseMatrix(rowNum,colNum);

		while(reader1.next(key1, value1)){
			int rowIdx1=key1.get();
			Vector rowVector1=value1.get().minus(meanVector);
			origin.assignRow(rowIdx1,rowVector1);
		}


		Matrix resu=origin.times(emat);
		
		try{
			PrintWriter out = new PrintWriter("svdResult.txt");
			for(int i=0;i<resu.rowSize();i++){
				for(Vector.Element e:resu.viewRow(i).nonZeroes()){
					out.print(e.get()+" ");
				}
				out.println();
				}
				out.close();
			}catch(Exception e){
    			System.out.println(e.getMessage());
    	}

		
	}
}
