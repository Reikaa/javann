import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

import java.util.HashMap;

import problems.Problem;
import problems.XOR;
import problems.RealNumbers;

import algorithms.*;

public class TestNN extends JFrame implements ActionListener, Runnable, TrainerListener{
	int width = 1000;
	int height = 500;

	int numHidden = 4;

	double minError = 0.05;

	int maxTrainers = 0;
	int numRunning = 0;

	Problem problem = null;

	BackProp backProp = null;
	QuickProp quickProp = null;
	GA ga = null;
	Pso pso = null;

	JButton runButton = new JButton("Run");
	JButton stopButton = new JButton("Stop");

	JTextField numHiddenText = new JTextField();
	JTextField minErrorText = new JTextField();

	JCheckBox backPropRun = new JCheckBox("Run", true);
	JTextField learningRateText = new JTextField();
	JTextField momentumText = new JTextField();

	JCheckBox quickPropRun = new JCheckBox("Run", true);
	JTextField qpMomentumText = new JTextField();

	JCheckBox gaRun = new JCheckBox("Run", true);
	JTextField gaPopSize = new JTextField();
	JTextField gaMutationRate = new JTextField();
	JTextField gaCrossoverRate = new JTextField();

	JCheckBox psoRun = new JCheckBox("Run", true);
	JTextField psoNumAgents = new JTextField();
	JTextField psoWeight = new JTextField();
	JTextField psoMomentum = new JTextField();
	JTextField psoMaxVelocity = new JTextField();

	DefaultListModel bpListModel = new DefaultListModel();
	JList bpOutputList = new JList(bpListModel);

	DefaultListModel qpListModel = new DefaultListModel();
	JList qpOutputList = new JList(qpListModel);

	DefaultListModel gaListModel = new DefaultListModel();
	JList gaOutputList = new JList(gaListModel);

	DefaultListModel psoListModel = new DefaultListModel();
	JList psoOutputList = new JList(psoListModel);

	HashMap labelMap = new HashMap();

	void init(){
		numHiddenText.setText(Integer.toString(numHidden));
		minErrorText.setText(Double.toString(minError));

		learningRateText.setText(Double.toString(BackProp.DEFAULT_LEARNING_RATE));
		momentumText.setText(Double.toString(BackProp.DEFAULT_MOMENTUM));

		qpMomentumText.setText(Double.toString(QuickProp.DEFAULT_MOMENTUM));

		gaPopSize.setText(Integer.toString(GA.MAX_POP));
		gaMutationRate.setText(Double.toString(GA.MUTATION_RATE));
		gaCrossoverRate.setText(Double.toString(GA.CROSSOVER_RATE));

		psoNumAgents.setText(Integer.toString(Pso.MAX_AGENTS));
		psoWeight.setText(Double.toString(Pso.DEFAULT_WEIGHT));
		psoMomentum.setText(Double.toString(Pso.DEFAULT_MOMENTUM));
		psoMaxVelocity.setText(Double.toString(Pso.DEFAULT_MAXVELOCITY));
	}

	public TestNN(){
		setTitle("NN Training Test");
		setSize(new Dimension(width, height));
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		JPanel panel = new JPanel();
		runButton.addActionListener(this);
		panel.add(runButton);
		stopButton.addActionListener(this);
		stopButton.setEnabled(false);
		panel.add(stopButton);
		getContentPane().add(panel, BorderLayout.SOUTH);

		int textWidth = 100;
		int textHeight = 21;

		numHiddenText.setPreferredSize(new Dimension(textWidth, textHeight));
		minErrorText.setPreferredSize(new Dimension(textWidth, textHeight));

		learningRateText.setPreferredSize(new Dimension(textWidth, textHeight));
		momentumText.setPreferredSize(new Dimension(textWidth, textHeight));

		qpMomentumText.setPreferredSize(new Dimension(textWidth, textHeight));

		gaPopSize.setPreferredSize(new Dimension(textWidth, textHeight));
		gaMutationRate.setPreferredSize(new Dimension(textWidth, textHeight));
		gaCrossoverRate.setPreferredSize(new Dimension(textWidth, textHeight));

		psoNumAgents.setPreferredSize(new Dimension(textWidth, textHeight));
		psoWeight.setPreferredSize(new Dimension(textWidth, textHeight));
		psoMomentum.setPreferredSize(new Dimension(textWidth, textHeight));
		psoMaxVelocity.setPreferredSize(new Dimension(textWidth, textHeight));

		panel = new JPanel();
		JPanel subPanel = new JPanel();
		subPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
		subPanel.add(new JLabel("Hidden Layer Size:"));
		subPanel.add(numHiddenText);
		panel.add(subPanel);
		subPanel = new JPanel();
		subPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
		subPanel.add(new JLabel("Min Error:"));
		subPanel.add(minErrorText);
		panel.add(subPanel);
		getContentPane().add(panel, BorderLayout.NORTH);

		JPanel centerPanel = new JPanel();
		centerPanel.setLayout(new GridLayout(1, 4));

		panel = new JPanel();
		panel.setBorder(BorderFactory.createTitledBorder("BackProp"));
		GridBagLayout gridBag = new GridBagLayout();
		panel.setLayout(gridBag);
		GridBagConstraints constraints = new GridBagConstraints();
		constraints.weightx = 1.0;
		constraints.fill = GridBagConstraints.BOTH;
		constraints.gridwidth = GridBagConstraints.REMAINDER;
		gridBag.setConstraints(backPropRun, constraints);
		panel.add(backPropRun);
		subPanel = new JPanel();
		subPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
		subPanel.add(new JLabel("Learning Rate:"));
		subPanel.add(learningRateText);
		gridBag.setConstraints(subPanel, constraints);
		panel.add(subPanel);
		subPanel = new JPanel();
		subPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
		subPanel.add(new JLabel("Momentum:"));
		subPanel.add(momentumText);
		gridBag.setConstraints(subPanel, constraints);
		panel.add(subPanel);
		JScrollPane scrollPane = new JScrollPane();
		scrollPane.getViewport().add(bpOutputList);
		constraints.gridheight = GridBagConstraints.REMAINDER;
		constraints.weighty = 1.0;
		gridBag.setConstraints(scrollPane, constraints);
		panel.add(scrollPane);

		labelMap.put(new Integer(Trainer.BACKPROP), bpListModel);

		centerPanel.add(panel);

		panel = new JPanel();
		panel.setBorder(BorderFactory.createTitledBorder("QuickProp"));
		gridBag = new GridBagLayout();
		panel.setLayout(gridBag);
		constraints.gridheight = 1;
		constraints.weighty = 0.0;
		gridBag.setConstraints(quickPropRun, constraints);
		panel.add(quickPropRun);
		subPanel = new JPanel();
		subPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
		subPanel.add(new JLabel("Momentum:"));
		subPanel.add(qpMomentumText);
		gridBag.setConstraints(subPanel, constraints);
		panel.add(subPanel);
		scrollPane = new JScrollPane();
		scrollPane.getViewport().add(qpOutputList);
		constraints.gridheight = GridBagConstraints.REMAINDER;
		constraints.weighty = 1.0;
		gridBag.setConstraints(scrollPane, constraints);
		panel.add(scrollPane);

		labelMap.put(new Integer(Trainer.QUICKPROP), qpListModel);

		centerPanel.add(panel);

		panel = new JPanel();
		panel.setBorder(BorderFactory.createTitledBorder("GA"));
		gridBag = new GridBagLayout();
		panel.setLayout(gridBag);
		constraints.gridheight = 1;
		constraints.weighty = 0.0;
		gridBag.setConstraints(gaRun, constraints);
		panel.add(gaRun);
		subPanel = new JPanel();
		subPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
		subPanel.add(new JLabel("Population Size:"));
		subPanel.add(gaPopSize);
		gridBag.setConstraints(subPanel, constraints);
		panel.add(subPanel);
		subPanel = new JPanel();
		subPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
		subPanel.add(new JLabel("Mutation Rate:"));
		subPanel.add(gaMutationRate);
		gridBag.setConstraints(subPanel, constraints);
		panel.add(subPanel);
		subPanel = new JPanel();
		subPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
		subPanel.add(new JLabel("Crossover Rate:"));
		subPanel.add(gaCrossoverRate);
		gridBag.setConstraints(subPanel, constraints);
		panel.add(subPanel);
		scrollPane = new JScrollPane();
		scrollPane.getViewport().add(gaOutputList);
		constraints.gridheight = GridBagConstraints.REMAINDER;
		constraints.weighty = 1.0;
		gridBag.setConstraints(scrollPane, constraints);
		panel.add(scrollPane);

		labelMap.put(new Integer(Trainer.GA), gaListModel);

		centerPanel.add(panel);

		panel = new JPanel();
		panel.setBorder(BorderFactory.createTitledBorder("PSO"));
		gridBag = new GridBagLayout();
		panel.setLayout(gridBag);
		constraints.gridheight = 1;
		constraints.weighty = 0.0;
		gridBag.setConstraints(psoRun, constraints);
		panel.add(psoRun);
		subPanel = new JPanel();
		subPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
		subPanel.add(new JLabel("Num Agents:"));
		subPanel.add(psoNumAgents);
		gridBag.setConstraints(subPanel, constraints);
		panel.add(subPanel);
		subPanel = new JPanel();
		subPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
		subPanel.add(new JLabel("Weight:"));
		subPanel.add(psoWeight);
		gridBag.setConstraints(subPanel, constraints);
		panel.add(subPanel);
		subPanel = new JPanel();
		subPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
		subPanel.add(new JLabel("Momentum:"));
		subPanel.add(psoMomentum);
		gridBag.setConstraints(subPanel, constraints);
		panel.add(subPanel);
		subPanel = new JPanel();
		subPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
		subPanel.add(new JLabel("Max Velocity:"));
		subPanel.add(psoMaxVelocity);
		gridBag.setConstraints(subPanel, constraints);
		panel.add(subPanel);
		scrollPane = new JScrollPane();
		scrollPane.getViewport().add(psoOutputList);
		constraints.gridheight = GridBagConstraints.REMAINDER;
		constraints.weighty = 1.0;
		gridBag.setConstraints(scrollPane, constraints);
		panel.add(scrollPane);

		labelMap.put(new Integer(Trainer.PSO), psoListModel);

		centerPanel.add(panel);

		getContentPane().add(centerPanel, BorderLayout.CENTER);

		init();
	}

	public void actionPerformed(ActionEvent e){
		Object source = e.getSource();
		if(source == runButton)
			(new Thread(this)).start();
		else if(source == stopButton){
			if(backProp != null){
				backProp.kill();
				backProp = null;
			}

			if(quickProp != null){
				quickProp.kill();
				quickProp = null;
			}

			if(ga != null){
				ga.kill();
				ga = null;
			}

			if(pso != null){
				pso.kill();
				pso = null;
			}
		}
	}

	public void trainingGenerationComplete(NeuralNetwork nn, Trainer trainer){
		//problem class takes care of updating output lists
	}

	public void trainingBegin(Trainer trainer){
		if(numRunning == 0){
			runButton.setEnabled(false);
			stopButton.setEnabled(true);
		}

		numRunning++;
	}

	public void trainingEnd(Trainer trainer){
		numRunning--;

		if(numRunning == 0){
			runButton.setEnabled(true);
			stopButton.setEnabled(false);
		}
	}

	public void run(){
		numRunning = 0;

		numHidden = Integer.parseInt(numHiddenText.getText());
		minError = Double.parseDouble(minErrorText.getText());

		problem = new XOR(numHidden, minError, labelMap);
//		problem = new RealNumbers(numHidden, minError, labelMap);

		if(backPropRun.isSelected()){
			backProp = new BackProp(numHidden, problem.getInputs(), problem.getOutputs(), minError);
			backProp.setLearningRate(Double.parseDouble(learningRateText.getText()));
			backProp.setMomentum(Double.parseDouble(momentumText.getText()));
			backProp.addTrainerListener(this);
			backProp.addTrainerListener(problem);
			backProp.start();
		}

		if(quickPropRun.isSelected()){
			quickProp = new QuickProp(numHidden, problem.getInputs(), problem.getOutputs(), minError);
			quickProp.setMomentum(Double.parseDouble(momentumText.getText()));
			quickProp.addTrainerListener(this);
			quickProp.addTrainerListener(problem);
			quickProp.start();
		}

		if(gaRun.isSelected()){
			ga = new GA(numHidden, problem.getInputs(), problem.getOutputs(), minError);
			ga.setPopSize(Integer.parseInt(gaPopSize.getText()));
			ga.setMutationRate(Double.parseDouble(gaMutationRate.getText()));
			ga.setCrossoverRate(Double.parseDouble(gaCrossoverRate.getText()));
			ga.addTrainerListener(this);
			ga.addTrainerListener(problem);
			ga.start();
		}

		if(psoRun.isSelected()){
			pso = new Pso(numHidden, problem.getInputs(), problem.getOutputs(), minError);
			pso.setNumAgents(Integer.parseInt(psoNumAgents.getText()));
			pso.setWeight(Double.parseDouble(psoWeight.getText()));
			pso.setMomentum(Double.parseDouble(psoMomentum.getText()));
			pso.setMaxVelocity(Double.parseDouble(psoMaxVelocity.getText()));
			pso.addTrainerListener(this);
			pso.addTrainerListener(problem);
			pso.start();
		}
	}

	public static void main(String [] args){
		TestNN t = new TestNN();
		t.show();
	}
}

// vim:noet:ts=3:sw=3
