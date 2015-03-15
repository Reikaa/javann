package problems;

import java.util.HashMap;
import java.util.Random;

import algorithms.Trainer;
import algorithms.TrainerListener;
import algorithms.NeuralNetwork;

public class Problem implements TrainerListener{
	protected int numPatterns = 0;
	protected double [][] inputs = null;
	protected double [][] outputs = null;
	protected int numHidden = 0;
	protected double minError = 0;

	protected HashMap outputLists = null;
	protected Random random = new Random(System.currentTimeMillis());

	public Problem(int numHidden, double minError, HashMap outputLists){
		this.numHidden = numHidden;
		this.minError = minError;
		this.outputLists = outputLists;
	}

	public int getNumPatterns(){
		return numPatterns;
	}

	public double[][] getInputs(){
		return inputs;
	}

	public double[][] getOutputs(){
		return outputs;
	}

	public double getMinError(){
		return minError;
	}

	public void trainingBegin(Trainer trainer){
	}

	public void trainingEnd(Trainer trainer){
	}

	public void trainingGenerationComplete(NeuralNetwork nn, Trainer trainer){
	}
}

// vim:noet:ts=3:sw=3
