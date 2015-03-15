package algorithms;

import java.util.Random;
import java.util.Arrays;

public class GA extends Trainer{
	public static final int    MAX_POP        =  100;
	public static final double MUTATION_RATE  =  0.1;
	public static final double CROSSOVER_RATE =  1.0;

	private int popSize = MAX_POP;
	private double mutationRate = MUTATION_RATE;
	private double crossoverRate = CROSSOVER_RATE;

	private NeuralNetwork population[] = null;

	public GA(int numHidden, double [][] inputs, double [][] targets, double minError){
		super(numHidden, inputs, targets, minError);
	}

	public int getType(){
		return Trainer.GA;
	}

	public void setPopSize(int popSize){
		this.popSize = popSize;
	}

	public void setMutationRate(double mutationRate){
		this.mutationRate = mutationRate;
	}

	public void setCrossoverRate(double crossoverRate){
		this.crossoverRate = crossoverRate;
	}

	public void mutate(NeuralNetwork net){
		double [][] inWeights = net.getInWeights();
		for(int i = 0; i < numInput; i++)
			for(int j = 0; j < numHidden; j++)
				inWeights[i][j] += random.nextGaussian();

		double [][] outWeights = net.getOutWeights();
		for(int i = 0; i < numHidden; i++)
			for(int j = 0; j < numOutput; j++)
				outWeights[i][j] += random.nextGaussian();
	}

	public void onePointCrossover(NeuralNetwork net1, NeuralNetwork net2){
		int range = numInput * numHidden + numHidden * numOutput;
		int point = random.nextInt(range);

		int count = 0;

		double [][] in1 = net1.getInWeights();
		double [][] in2 = net2.getInWeights();
		for(int i = 0; i < numInput; i++){
			for(int j = 0; j < numHidden; j++, count++){
				if(count >= point)
					in1[i][j] = in2[i][j];
			}
		}

		double [][] out1 = net1.getOutWeights();
		double [][] out2 = net2.getOutWeights();
		for(int i = 0; i < numHidden; i++){
			for(int j = 0; j < numOutput; j++, count++){
				if(count >= point)
					out1[i][j] = out2[i][j];
			}
		}
	}

	public void twoPointCrossover(NeuralNetwork net1, NeuralNetwork net2){
		int range = numInput * numHidden + numHidden * numOutput;
		int start = random.nextInt(range);
		int end = random.nextInt(range - start) + start;

		int count = 0;

		double [][] in1 = net1.getInWeights();
		double [][] in2 = net2.getInWeights();
		for(int i = 0; i < numInput; i++){
			for(int j = 0; j < numHidden; j++, count++){
				if(count >= start && count <= end)
					in1[i][j] = in2[i][j];
			}
		}

		double [][] out1 = net1.getOutWeights();
		double [][] out2 = net2.getOutWeights();
		for(int i = 0; i < numHidden; i++){
			for(int j = 0; j < numOutput; j++, count++){
				if(count >= start && count <= end)
					out1[i][j] = out2[i][j];
			}
		}
	}

	public void uniformCrossover(NeuralNetwork net1, NeuralNetwork net2){
		double crossoverRate = 0.01;

		double [][] in1 = net1.getInWeights();
		double [][] in2 = net2.getInWeights();
		for(int i = 0; i < numInput; i++){
			for(int j = 0; j < numHidden; j++){
				if(random.nextDouble() < crossoverRate)
					in1[i][j] = in2[i][j];
			}
		}

		double [][] out1 = net1.getOutWeights();
		double [][] out2 = net2.getOutWeights();
		for(int i = 0; i < numHidden; i++){
			for(int j = 0; j < numOutput; j++){
				if(random.nextDouble() < crossoverRate)
					out1[i][j] = out2[i][j];
			}
		}
	}

	public void crossover(NeuralNetwork net1, NeuralNetwork net2){
//		onePointCrossover(net1, net2);
//		twoPointCrossover(net1, net2);
		uniformCrossover(net1, net2);
	}


	private void evolve(){
		NeuralNetwork newPop[] = new NeuralNetwork[popSize];

		int minIndex = 0;
		double minFitness = 1000.0;
		int maxIndex = 0;
		double maxFitness = 0;

		double sum = 0.0;

		for(int i = 0; i < popSize; i++){
			double fitness = population[i].getFitness();
			sum += fitness;
			if(fitness < minFitness){
				minFitness = fitness;
				minIndex = i;
			}
			if(fitness > maxFitness){
				maxFitness = fitness;
				maxIndex = i;
			}
		}

//		System.err.println("[" + minFitness + ", " + maxFitness + "]");

		double [] normals = new double[popSize];
		for(int i = 0; i < popSize; i++)
			normals[i] = (1.0 - population[i].getFitness()) / sum;

		//roulette wheel selection
		int numAdded = 0;
		while(numAdded < popSize){
			for(int i = 0; i < popSize; i++){
				double roll = random.nextDouble();
				while(roll <= normals[i] && numAdded < popSize){
					newPop[numAdded++] = population[i].copy();
					roll = random.nextDouble();
				}
			}
		}

		for(int i = 0; i < popSize; i++){
			//keep best individual
			if(i == minIndex)
				continue;

			population[i] = newPop[i];

			if(random.nextDouble() < crossoverRate)
				crossover(population[i], newPop[(i + 1) % popSize]);

			if(random.nextDouble() < mutationRate)
				mutate(population[i]);
		}
	}

	public void run(){
		broadcastBegin();

		population = new NeuralNetwork[popSize];

		for(int i = 0; i < popSize; i++)
			population[i] = new NeuralNetwork(numInput, numHidden, numOutput);

		double fitness = 20.0;
		while(fitness > minError && isRunning){
			numGenerations++;

			fitness = 10000.0;
			int best = 0;
			for(int i = 0; i < popSize; i++){
				double f = population[i].evaluate(inputs, targets);
				if(f < fitness){
					fitness = f;
					best = i;
				}
			}

			broadcastGenerationComplete(population[best]);

			evolve();
		}

		broadcastEnd();
	}
}

// vim:noet:ts=3:sw=3
