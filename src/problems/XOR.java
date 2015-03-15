package problems;

import javax.swing.DefaultListModel;

import java.util.HashMap;

import algorithms.Trainer;
import algorithms.TrainerListener;
import algorithms.NeuralNetwork;

public class XOR extends Problem{
	public XOR(int numHidden, double minError, HashMap outputLists){
		super(numHidden, minError, outputLists);

		this.outputLists = outputLists;

		numPatterns = 1000;
		inputs = new double[numPatterns][2];
		outputs = new double[numPatterns][1];
		for(int i = 0; i < numPatterns; i++){
			inputs[i][0] = random.nextInt(2);
			inputs[i][1] = random.nextInt(2);
			outputs[i][0] = (int)inputs[i][0] ^ (int)inputs[i][1];
		}
	}

	public String testPattern(double [] input, NeuralNetwork nn){
		int result = (int)input[0] ^ (int)input[1];

		double [] output = new double[1];
		nn.activate(input, output);

		double nnResult = 0;
		if(output[0] > 0.7)
			nnResult = 1;
		else if(output[0] > 0.3)
			nnResult = 0.5;

		String outputText = "[" + input[0] + ", " + input[1] + ", " + result + "] -> " + output[0] + " (" + nnResult + ")";

		return outputText;
	}

	public void trainingGenerationComplete(NeuralNetwork nn, Trainer trainer){
		if(nn != null && trainer != null){
			DefaultListModel listModel = (DefaultListModel)outputLists.get(new Integer(trainer.getType()));

			listModel.setSize(6);

			double [] input = new double[2];

			input[0] = 0; input[1] = 0;
			listModel.setElementAt(testPattern(input, nn), 0);

			input[0] = 0; input[1] = 1;
			listModel.setElementAt(testPattern(input, nn), 1);

			input[0] = 1; input[1] = 0;
			listModel.setElementAt(testPattern(input, nn), 2);

			input[0] = 1; input[1] = 1;
			listModel.setElementAt(testPattern(input, nn), 3);

			listModel.setElementAt(Double.toString(nn.getFitness()), 4);

			if(trainer != null)
				listModel.setElementAt(Integer.toString(trainer.getNumGenerations()), 5);
			else
				listModel.setElementAt("0", 5);
		}
	}
}

// vim:noet:ts=3:sw=3
