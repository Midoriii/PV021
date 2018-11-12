package pv021;

import java.util.ArrayList;

public class NeuralNetworksProject {
    public static void main(String [] args){

        //Number of input neurons, hidden layers
        int inputDimensions = 784;
        int trainImagesCount = 60000;
        int testImagesCount = 10000;
        int hiddenLayersCount = 2;

        //How many epochs
        int epochs = 24;

        //Numbers of neurons in each hidden layer
        ArrayList<Integer> hiddenLayersNeuronCount = new ArrayList<Integer>();
        hiddenLayersNeuronCount.add(24);
        hiddenLayersNeuronCount.add(36);


        //Create the network
        Network Net = new Network(inputDimensions, trainImagesCount, testImagesCount, hiddenLayersCount, hiddenLayersNeuronCount);


        //Read and pre-process the inputs, includes one hot mapping
        Net.readMnist();
        //Pre-processing
        Net.maxValueNormalization();
        //Net.zScoreNormalization();


        //Add cycle here
        //Compute the forward pass
        Net.forwardPass();



        //DEBUG
        /*

        System.out.println("Oi");
        for(int i = 0; i < Net.getHiddenLayers().get(0).getInputs().length; i++){
            System.out.println(Net.getHiddenLayers().get(0).getInputs()[i]);
        }
        System.out.println("Hi");
        for(int i = 0; i < Net.getOutputLayer().getInputs().length; i++){
            System.out.println(Net.getOutputLayer().getInputs()[i]);
        }
        System.out.println("Go");
        for(int i = 0; i < Net.getOutputLayer().getNeurons().get(3).getWeights().length; i++){
            System.out.println(Net.getOutputLayer().getNeurons().get(3).getWeights()[i]);
        }

        System.out.println("eh");
        System.out.println(Net.getOutputLayer().getOutputs()[2]);
        System.out.println(Net.getOutputLayer().getOutputs()[7]);
        System.out.println("sum:");
        double sum = 0.0;
        for(int i = 0; i < Net.getOutputLayer().getOutputs().length; i++){
            sum += Net.getOutputLayer().getOutputs()[i];
        }
        System.out.println(sum);

        for(int i = 0; i < Net.getTrainImages()[8].length; i++){
            System.out.println("i: " + i + " " + Net.getTrainImages()[8][i]);
        }

        System.out.println(Net.getTrainImages().length);
        System.out.println(Net.getTestImages().length);

        System.out.println(Net.getTrainLabels().length);
        System.out.println(Net.getTestLabels().length);

        for(int i = 0; i < Net.getTrainLabels().length; i++){
            System.out.println("i: " + i + " " + Net.getTrainLabels()[i]);
        }



        System.out.println(Net.getTestLabelsOneHot().length);
        System.out.println(Net.getTrainLabelsOneHot().length);
        System.out.println(Net.getTestLabelsOneHot()[2].length);
        
        for(int i = 0; i < Net.getTrainLabelsOneHot()[2].length; i++){
            System.out.println(Net.getTrainLabelsOneHot()[2][i]);
        }

        System.out.println();
        for(int i = 0; i < Net.getTrainLabelsOneHot()[1].length; i++){
            System.out.println(Net.getTrainLabelsOneHot()[1][i]);
        }
        System.out.println();
        for(int i = 0; i < Net.getTrainLabelsOneHot()[0].length; i++){
            System.out.println(Net.getTrainLabelsOneHot()[0][i]);
        }

        */

        //DEBUG

        System.out.println("success");
    }
}
