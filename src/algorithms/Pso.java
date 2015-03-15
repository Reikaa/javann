package algorithms;

import java.util.Random;

public class Pso extends Trainer{
	public static int MAX_AGENTS = 160;
	public static double DEFAULT_WEIGHT = 1.0;
	public static double DEFAULT_MOMENTUM = 1.0;
	public static double DEFAULT_MAXVELOCITY = 10.0;

	private int numAgents = MAX_AGENTS;
	private double weight = DEFAULT_WEIGHT;
	private double momentum = DEFAULT_MOMENTUM;
	private double maxVelocity = DEFAULT_MAXVELOCITY;

	public Pso(int numHidden, double [][] inputs, double [][] targets, double minError){
		super(numHidden, inputs, targets, minError);
	}

	public int getType(){
		return Trainer.PSO;
	}

	public void setNumAgents(int numAgents){
		this.numAgents = numAgents;
	}

	public void setWeight(double weight){
		this.weight = weight;
	}

	public void setMomentum(double momentum){
		this.momentum = momentum;
	}

	public void setMaxVelocity(double maxVelocity){
		this.maxVelocity = maxVelocity;
	}

	public void run(){
		broadcastBegin();

		int dimension = numInput * numHidden + numHidden * numOutput;

		NeuralNetwork [] population = new NeuralNetwork[numAgents];
		NeuralNetwork [] bestNet = new NeuralNetwork[numAgents];

		for(int i = 0; i < numAgents; i++){
			population[i] = new NeuralNetwork(numInput, numHidden, numOutput);
			bestNet[i] = population[i].copy();
		}

		double [][] inVelocities = new double[numInput][numHidden];
		for(int i = 0; i < numInput; i++){
			for(int j = 0; j < numHidden; j++){
				inVelocities[i][j] = random.nextDouble() * maxVelocity;
				if(random.nextDouble() > 0.5)
					inVelocities[i][j] = -inVelocities[i][j];
			}
		}

		double [][] outVelocities = new double[numHidden][numOutput];
		for(int i = 0; i < numHidden; i++){
			for(int j = 0; j < numOutput; j++){
				outVelocities[i][j] = random.nextDouble() * maxVelocity;
				if(random.nextDouble() > 0.5)
					outVelocities[i][j] = -outVelocities[i][j];
			}
		}

		double [] popBest = new double[numAgents];

		boolean firstTime = true; //first iteration of this run
		int best = 0;        //initialy assume the first particle as the best
		boolean finish = false;

		while(!finish && isRunning){
			numGenerations++;

			NeuralNetwork [] nextPop = new NeuralNetwork[numAgents];

			for(int i = 0; i < numAgents; i++){
				double fitness = population[i].evaluate(inputs, targets);

				if(firstTime)
					popBest[i] = fitness;
				else if(fitness < popBest[i]){
					popBest[i] = fitness;

					bestNet[i] = population[i].copy();

					if(fitness < popBest[best])
						best = i;
				}

				nextPop[i] = population[i].copy();

				/* asynchronous version */
				adjustWeights(nextPop[i].getInWeights(), inVelocities, bestNet[i].getInWeights(), bestNet[best].getInWeights());
				adjustWeights(nextPop[i].getOutWeights(), outVelocities, bestNet[i].getOutWeights(), bestNet[best].getOutWeights());
			}

			for(int i = 0; i < numAgents; i++)
				population[i] = nextPop[i];

			weight *= momentum;

			bestNet[best].evaluate(inputs, targets);
			broadcastGenerationComplete(bestNet[best]);

			finish = (popBest[best] <= minError);

			firstTime = false;
		}

		broadcastEnd();
	}

	private void adjustWeights(double [][] weights, double [][] velocity, double [][] agentBest, double [][] popBest){
		int size1 = weights.length;
		int size2 = weights[0].length;

		for(int i = 0; i < size1; i++){
			for(int j = 0; j < size2; j++){
				double weightValue = weights[i][j];
				double currentBestWeight = agentBest[i][j];
				double popBestWeight = popBest[i][j];

				velocity[i][j] = weight * velocity[i][j] + 2 * random.nextDouble() * (currentBestWeight - weightValue) + 2 * random.nextDouble() * (popBestWeight - weightValue);
				if(velocity[i][j] > maxVelocity)
					velocity[i][j] = maxVelocity;
				else if(velocity[i][j] < -maxVelocity)
					velocity[i][j] = -maxVelocity;

				weights[i][j] += velocity[i][j];
			}
		}
	}
}

// vim:noet:ts=3:sw=3
