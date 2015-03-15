package algorithms;

public class BackProp extends Trainer{
	public static final double DEFAULT_LEARNING_RATE = 0.9;
	public static final double DEFAULT_MOMENTUM = 1.0;

	private double learningRate = DEFAULT_LEARNING_RATE; // [0.0-100.0]
	private double momentum = DEFAULT_MOMENTUM; // [0.0-1.0]

	private NeuralNetwork nn = null;

	public BackProp(NeuralNetwork nn, double [][] inputs, double [][] targets, double minError){
		super(nn.getNumHidden(), inputs, targets, minError);
		this.nn = nn;
	}

	public BackProp(int numHidden, double [][] inputs, double [][] targets, double minError){
		super(numHidden, inputs, targets, minError);
		nn = new NeuralNetwork(numInput, numHidden, numOutput);
	}

	public int getType(){
		return Trainer.BACKPROP;
	}

	public double getLearningRate(){
		return learningRate;
	}

	public void setLearningRate(double learningRate){
		this.learningRate = learningRate;
	}

	public double getMomentum(){
		return momentum;
	}

	public void setMomentum(double momentum){
		this.momentum = momentum;
	}

	public void run(){
		broadcastBegin();

		double fitness = 20.0;
		while(fitness > minError && isRunning){
			numGenerations++;
			fitness = 0;
			for(int i = 0; i < numPatterns; i++)
				fitness += adjustWeights(inputs[i], targets[i]);
			fitness /= numPatterns;
			learningRate *= momentum;

			nn.setFitness(fitness);

			broadcastGenerationComplete(nn);
		}

		broadcastEnd();
	}

	private double adjustWeights(double [] inputs, double [] targets){
		//activate network
		double [] hidden = new double[numHidden];
		double [] outputs = new double[numOutput];

		nn.activate(inputs, hidden, outputs);

		//calculate output delta
		double [] outDelta = new double[numOutput];

		for(int i = 0; i < numOutput; i++)
			outDelta[i] = (targets[i] - outputs[i]) * outputs[i] * (1.0 - outputs[i]);
		double [][] outWeights = nn.getOutWeights();

		//update output layer
		for(int i = 0; i < numHidden; i++){
			for(int j = 0; j < numOutput; j++)
				outWeights[i][j] += learningRate * outDelta[j] * hidden[i];
		}

		double [][] inWeights = nn.getInWeights();

		//update input layer
		for(int i = 0; i < numHidden; i++){
			double sum = 0.0;
			for(int j = 0; j < numOutput; j++)
				sum += outDelta[j] * outWeights[i][j];
			double hiddenDelta = sum * hidden[i] * (1.0 - hidden[i]);
			for(int j = 0; j < numInput; j++)
				inWeights[j][i] += learningRate * hiddenDelta * inputs[j];
		}

		//caculate fitness for run
		return NeuralNetwork.sumSquaredError(outputs, targets);
	}
}

// vim:noet:ts=3:sw=3
