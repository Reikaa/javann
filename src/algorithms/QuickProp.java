package algorithms;

public class QuickProp extends Trainer{
	public static final double DEFAULT_MOMENTUM = 0.9;

	private double momentum = DEFAULT_MOMENTUM; // [0.0-1.0]

	// Inside thresh, do grad descent; outside, jump.
	private double modeSwitchThreshold = 0.0;
	// Don't jump more than this times last step
	private double maxFactor = 1.75;
	private double shrinkFactor = maxFactor / (1.0 + maxFactor);
	// divide epsilon by fan-in before use
	private boolean splitEpsilon = false; //true;
	// For grad descent if last step was (almost) 0
	private double epsilon = 0.55; /* 1.0 */
	// Weight decay
	private double decay = -0.0001;

	private double [][] prevInSlopes = null;
	private double [][] inSlopes = null;
	private double [][] inDeltaWeights = null;

	private double [][] prevOutSlopes = null;
	private double [][] outSlopes = null;
	private double [][] outDeltaWeights = null;

	private NeuralNetwork nn = null;

	public QuickProp(NeuralNetwork nn, double [][] inputs, double [][] targets, double minError){
		super(nn.getNumHidden(), inputs, targets, minError);
		this.nn = nn;
	}

	public QuickProp(int numHidden, double [][] inputs, double [][] targets, double minError){
		super(numHidden, inputs, targets, minError);
		nn = new NeuralNetwork(numInput, numHidden, numOutput);
	}

	public int getType(){
		return Trainer.QUICKPROP;
	}

	public void setMomentum(double momentum){
		this.momentum = momentum;
	}

	public void run(){
		broadcastBegin();

		prevInSlopes = new double[numInput][numHidden];
		inSlopes = new double[numInput][numHidden];
		inDeltaWeights = new double[numInput][numHidden];
		for(int i = 0; i < numInput; i++)
			for(int j = 0; j < numHidden; j++)
				inDeltaWeights[i][j] = prevInSlopes[i][j] = inSlopes[i][j] = 0.0;

		prevOutSlopes = new double[numHidden][numOutput];
		outSlopes = new double[numHidden][numOutput];
		outDeltaWeights = new double[numHidden][numOutput];
		for(int i = 0; i < numHidden; i++)
			for(int j = 0; j < numOutput; j++)
				outDeltaWeights[i][j] = prevOutSlopes[i][j] = outSlopes[i][j] = 0.0;

		double fitness = 1000.0;

		while(fitness > minError && isRunning){
			numGenerations++;

			updateSlopes(inSlopes, prevInSlopes, nn.getInWeights());
			updateSlopes(outSlopes, prevOutSlopes, nn.getOutWeights());

			fitness = 0.0;
			for(int i = 0; i < numPatterns; i++)
				fitness += adjustWeights(inputs[i], targets[i]);
			fitness /= numPatterns;

			nn.setFitness(fitness);

			broadcastGenerationComplete(nn);
		}

		broadcastEnd();
	}

	private void updateSlopes(double [][] slopes, double [][] prevSlopes, double [][] weights){
		int size1 = slopes.length;
		int size2 = slopes[0].length;
		for(int i = 0; i < size1; i++){
			for(int j = 0; j < size2; j++){
				prevSlopes[i][j] = slopes[i][j];
				slopes[i][j] = decay * weights[i][j];
			}
		}
	}

	private double adjustWeights(double [] inputs, double [] targets){
		double [] hidden = new double[numHidden];
		double [] outputs = new double[numOutput];

		nn.activate(inputs, hidden, outputs);

		double [] outError = new double[numOutput];

		for(int i = 0; i < numOutput; i++)
			outError[i] = (targets[i] - outputs[i]) * outputs[i] * (1.0 - outputs[i]);

		double [][] outWeights = nn.getOutWeights();
		double [] hiddenError = new double[numHidden];

		for(int i = 0; i < numHidden; i++){
			double sum = 0.0;
			for(int j = 0; j < numOutput; j++)
				sum += outError[j] * outWeights[i][j];
			hiddenError[i] = sum * hidden[i] * (1.0 - hidden[i]);
		}

		for(int i = 0; i < numInput; i++)
			for(int j = 0; j < numHidden; j++)
				inSlopes[i][j] += hiddenError[j] * hidden[j];

		for(int i = 0; i < numHidden; i++)
			for(int j = 0; j < numOutput; j++)
				outSlopes[i][j] += outError[j] * outputs[j];

		takeStep(nn.getInWeights(), inDeltaWeights, inSlopes, prevInSlopes);
		takeStep(outWeights, outDeltaWeights, outSlopes, prevOutSlopes);

		return NeuralNetwork.sumSquaredError(outputs, targets);
	}

	private void takeStep(double [][] weights, double [][] deltaWeights, double [][] slopes, double [][] prevSlopes){
		int size1 = weights.length;
		int size2 = weights[0].length;

		for(int i = 0; i < size1; i++){
			for(int j = 0; j < size2; j++){
				double nextStep = 0.0;

				if(deltaWeights[i][j] > modeSwitchThreshold){
					if(slopes[i][j] > 0.0)
						nextStep = (splitEpsilon ? ((epsilon * slopes[i][j]) / size1)
						                         : (epsilon * slopes[i][j]));

					if(slopes[i][j] > (shrinkFactor * prevSlopes[i][j]))
						nextStep += maxFactor * deltaWeights[i][j];
					else
						nextStep += (slopes[i][j] / (prevSlopes[i][j] - slopes[i][j])) * deltaWeights[i][j];
				}
				else if(deltaWeights[i][j] < -modeSwitchThreshold){
					if(slopes[i][j] < 0.0)
						nextStep = (splitEpsilon ? ((epsilon * slopes[i][j]) / size1)
						                         : (epsilon * slopes[i][j]));

					if(slopes[i][j] < (shrinkFactor * prevSlopes[i][j]))
						nextStep += maxFactor * deltaWeights[i][j];
					else
						nextStep += (slopes[i][j] / (prevSlopes[i][j] - slopes[i][j])) * deltaWeights[i][j];
				}
				else{
					nextStep = (splitEpsilon ? ((epsilon * slopes[i][j]) / size1)
					                         : (epsilon * slopes[i][j]))
					           + (momentum * deltaWeights[i][j]);
				}

//				System.out.print(slopes[i][j] + "," + nextStep);
//				System.out.print("," + deltaWeights[i][j] + "," + weights[i][j]);
				deltaWeights[i][j] = nextStep;
				weights[i][j] += nextStep;
//				System.out.print("," + deltaWeights[i][j] + "," + weights[i][j]);
//				System.out.println("");
			}
		}
	}
}

// vim:noet:ts=3:sw=3
