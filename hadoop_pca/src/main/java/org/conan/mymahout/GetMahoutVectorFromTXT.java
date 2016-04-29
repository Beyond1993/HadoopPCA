package org.conan.mymahout;

import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.StringTokenizer;
import java.util.ArrayList;
import java.net.URI;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.DenseVector;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import java.io.DataInputStream;

public class GetMahoutVectorFromTXT{
	private static int rowNum,colNum;
	public static ArrayList<Integer> readTXTWriteMahoutVector(String input,String output) throws Exception{
		Configuration config = new Configuration();
		//for laocal hdfs
		FileSystem fs=FileSystem.get(config);
		FSDataInputStream in = fs.open(new Path(input));
		
		// Open a file for reading
//		FileSystem fs = FileSystem.get(URI.create(input),config);		
//		Path cFile = new Path(input);  
//		DataInputStream in = new DataInputStream(fs.open(cFile));

		BufferedReader reader =new BufferedReader(new InputStreamReader(in));
		SequenceFile.Writer writer = SequenceFile.createWriter(fs ,config,new Path(output),IntWritable.class,VectorWritable.class);
		String line;
		int lineNum=0;
		while((line=reader.readLine())!=null){
			writer.append(new IntWritable(lineNum++),new VectorWritable(StringToMahoutVector(line)));
		}
		rowNum=lineNum;
		ArrayList<Integer> list=new ArrayList<Integer>();
		list.add(rowNum);
		list.add(colNum);
		writer.close();
		reader.close();
		return list;
	}
	private static DenseVector StringToMahoutVector(String line){
		StringTokenizer tokenizer = new StringTokenizer(line);
		colNum=tokenizer.countTokens();
		DenseVector mahoutVector = new DenseVector(colNum);
		int i=0;
		while(tokenizer.hasMoreTokens())
			mahoutVector.set(i++,Double.parseDouble(tokenizer.nextToken()));
		return mahoutVector;
		
	}
	public static void main(String[] args) throws Exception{
		readTXTWriteMahoutVector(args[0],args[1]);
	}
}
