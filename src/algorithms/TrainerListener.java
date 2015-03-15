package algorithms;

public interface TrainerListener{
	public void trainingBegin(Trainer trainer);
	public void trainingEnd(Trainer trainer);
	public void trainingGenerationComplete(NeuralNetwork nn, Trainer trainer);
}

// vim:noet:ts=3:sw=3
