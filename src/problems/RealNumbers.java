package problems;

import javax.swing.DefaultListModel;

import java.util.HashMap;

import algorithms.Trainer;
import algorithms.TrainerListener;
import algorithms.NeuralNetwork;

public class RealNumbers extends Problem{
	private int numBits = 4;
	private int numInputNumbers = 2;
	private int upperBound = (int)Math.pow(2, numBits) - 1;

	private int numTestCases = 4;

	// calculate sum of numInputNumbers numbers...
	public RealNumbers(int numHidden, double minError, HashMap outputLists){
		super(numHidden, minError, outputLists);

		numPatterns = 1000;

		inputs = new double[numPatterns][numBits * numInputNumbers];
		outputs = new double[numPatterns][numBits];

		for(int i = 0; i < numPatterns; i++){
			int sum = 0;

			for(int j = 0; j < numInputNumbers; j++){
				int num = random.nextInt(upperBound);
				fillBits(num, inputs[i], j * numBits);
				sum += num;
			}

			sum %= upperBound;
			fillBits(sum, outputs[i], 0);
		}
	}

	private void fillBits(int num, double [] inputs, int start){
		int end = start + numBits;
		int shiftBit = 1;
		for(int i = start; i < end; i++){
			inputs[i] = (num & shiftBit) > 0 ? 1 : 0;
			shiftBit <<= 1;
		}
	}

	private int getBits(double [] outputs, int start){
		int num = 0;
		int shiftBit = 1;
		int end = start + numBits;
		for(int i = start; i < end; i++){
			if(outputs[i] > 0.7) //cutoff for value of 1
				num |= shiftBit;
			shiftBit <<= 1;
		}
		return num;
	}

	public int testPattern(double [] input, NeuralNetwork nn){
		double [] output = new double[numBits];
		nn.activate(input, output);
		return getBits(output, 0);
	}

	public void trainingGenerationComplete(NeuralNetwork nn, Trainer trainer){
		if(nn != null && trainer != null){
			DefaultListModel listModel = (DefaultListModel)outputLists.get(new Integer(trainer.getType()));

			listModel.setSize(numTestCases + 2);

			double [] input = new double[numBits * numInputNumbers];

			for(int i = 0; i < numTestCases; i++){
				int sum = 0;

				String outputText = "[";

				for(int j = 0; j < numInputNumbers; j++){
					int num = random.nextInt(upperBound);
					if(j > 0)
						outputText += ", ";
					outputText += num;
					fillBits(num, input, j * numBits);
					sum += num;
				}

				sum %= upperBound;
				outputText += "] -> " + testPattern(input, nn) + "[" + sum + "]";

				listModel.setElementAt(outputText, i);
			}

			int index = numTestCases;
			listModel.setElementAt(Double.toString(nn.getFitness()), index++);

			if(trainer != null)
				listModel.setElementAt(Integer.toString(trainer.getNumGenerations()), index++);
			else
				listModel.setElementAt("0", index++);
		}
	}
}

// vim:noet:ts=3:sw=3
