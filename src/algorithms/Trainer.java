package algorithms;

import java.util.Vector;
import java.util.Random;

public abstract class Trainer extends Thread{
	public static final int NONE = 0;
	public static final int BACKPROP = 1;
	public static final int QUICKPROP = 2;
	public static final int GA = 3;
	public static final int PSO = 4;

	int numGenerations;
	int numPatterns;

	int numInput, numHidden, numOutput;

	double [][] inputs;
	double [][] targets;

	double minError;

	boolean isRunning = false;

	Vector trainerListeners = new Vector();

	Random random = new Random(System.currentTimeMillis());

	public Trainer(int numHidden, double [][] inputs, double [][] targets, double minError){
		numGenerations = 0;
		numPatterns = inputs.length;

		if(numPatterns > 0){
			numInput = inputs[0].length;
			this.numHidden = numHidden;
			numOutput = targets[0].length;

			this.inputs = inputs;
			this.targets = targets;

			this.minError = minError;
		}
		else
			throw (new NullPointerException("No Patterns Supplied"));
	}

	public void kill(){
		isRunning = false;
	}

	public abstract void run();

	public void addTrainerListener(TrainerListener listener){
		trainerListeners.addElement(listener);
	}

	public void broadcastBegin(){
		isRunning = true;

		int numListeners = trainerListeners.size();
		for(int i = 0; i < numListeners; i++)
			((TrainerListener)trainerListeners.elementAt(i)).trainingBegin(this);
	}

	public void broadcastEnd(){
		isRunning = false;

		int numListeners = trainerListeners.size();
		for(int i = 0; i < numListeners; i++)
			((TrainerListener)trainerListeners.elementAt(i)).trainingEnd(this);
	}

	public void broadcastGenerationComplete(NeuralNetwork nn){
		int numListeners = trainerListeners.size();
		for(int i = 0; i < numListeners; i++)
			((TrainerListener)trainerListeners.elementAt(i)).trainingGenerationComplete(nn, this);
	}

	public int getNumGenerations(){
		return numGenerations;
	}

	public abstract int getType();
}

// vim:noet:ts=3:sw=3
