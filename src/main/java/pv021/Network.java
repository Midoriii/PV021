package pv021;

import lombok.Getter;
import lombok.Setter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

@Getter
@Setter
public class Network {

    private double momentum = 0.5;
    private double learningRate = 0.01;
    private int categories = 10;

    private Layer inputLayer;
    private ArrayList<Layer> hiddenLayers = new ArrayList<Layer>();
    private Layer outputLayer;

    private double[][] trainImages = new double[60000][784];
    private double[][] testImages = new double[10000][784];
    private int[] trainLabels = new int[60000];
    private int[] testLabels = new int[10000];
    private int[][] trainLabelsOneHot = new int[60000][10];
    private int[][] testLabelsOneHot = new int[10000][10];

    private int[] trainPredictions = new int[60000];
    private int[] testPredictions = new int[10000];


    //Init layers in constructor
    public Network(int inputLayerInputs, int hiddenLayersCount, ArrayList<Integer> hiddenLayersNeuronCount){
        //Create desired layers
        inputLayer = new Layer(1, inputLayerInputs);
        for(int i = 0; i < hiddenLayersCount; i++){
            //First hidden layer has the weights number equal to the neurons in input layer
            if(i == 0){
                hiddenLayers.add(new Layer(inputLayerInputs, hiddenLayersNeuronCount.get(i)));
            }
            else{
                hiddenLayers.add(new Layer(hiddenLayersNeuronCount.get(i-1), hiddenLayersNeuronCount.get(i)));
            }
            //Output layer has the number of weights equal to the number of neurons in the last hidden layer
        outputLayer = new Layer(hiddenLayersNeuronCount.get(hiddenLayersNeuronCount.size() - 1), categories);
        }
    }

    //Forward pass
    public void forwardPass(){
        //Inputs of the input layer are its own input .. + 1.0 for bias
        inputLayer.setInputs(getInput());
        inputLayer.setOutputs(inputLayer.getInputs());

        //Pass the previous outputs into current inputs, calculate activation, pass the outputs
        for(int i = 0; i < hiddenLayers.size(); i++){

            //Set the inputs to the previous layer's outputs
            if(i == 0){
                hiddenLayers.get(i).setInputs(inputLayer.getOutputs());
            }
            else{
                hiddenLayers.get(i).setInputs(hiddenLayers.get(i-1).getOutputs());
            }

            //Calculate new output
            hiddenLayers.get(i).calculateOutputs("Relu");
        }

        //Calculate output Layer outputs
        outputLayer.setInputs(hiddenLayers.get(hiddenLayers.size()-1).getOutputs());
        outputLayer.calculateOutputs("Softmax");
    }

    //Stub to feed image vectors to the forward pass
    private double[] getInput(){
        double[] results = new double[784];
        for(int i = 0; i < inputLayer.getNeurons().size(); i++){
            results[i] = 2.5;
        }
        return results;
    }

    //Read train and test image vectors and labels
    public void readMnist(){
        String mnistTrainVectors = "mnist/mnist_train_vectors.csv";
        String mnistTestVectors = "mnist/mnist_test_vectors.csv";
        String mnistTrainLabels = "mnist/mnist_train_labels.csv";
        String mnistTestLabels = "mnist/mnist_test_labels.csv";

        trainImages = parseVectors(mnistTrainVectors, "train");
        testImages = parseVectors(mnistTestVectors, "test");
        trainLabels = parseLabels(mnistTrainLabels, "train");
        testLabels = parseLabels(mnistTestLabels, "test");
        trainLabelsOneHot = mapOneHotLabels(trainLabels, "train");
        testLabelsOneHot = mapOneHotLabels(testLabels, "test");
    }

    //Parse those .csv files
    private double[][] parseVectors(String path, String type){
        String line;
        String cvsSplitBy = ",";
        double[][] vector;
        if(type != "test"){
            vector = new double[60000][784];
        }
        else{
            vector = new double[10000][784];
        }

        //Get file from resources folder
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource(path).getFile());

        try (Scanner scanner = new Scanner(file)) {
            int j = 0;

            while (scanner.hasNextLine()) {
                //Read line
                line = scanner.nextLine();
                //Split the line
                String[] addition = line.split(cvsSplitBy);

                //Parse strings into doubles and add the single image into the set
                for(int i = 0; i < addition.length; i++){
                    vector[j][i] = (Double.parseDouble(addition[i]));
                }

                j++;
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return vector;
    }

    private int[] parseLabels(String path, String type){
        String line;
        int[] vector;
        if(type != "test"){
            vector = new int[60000];
        }
        else{
            vector = new int[10000];
        }

        //Get file from resources folder
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource(path).getFile());

        try (Scanner scanner = new Scanner(file)) {
            int i = 0;
            while (scanner.hasNextLine()) {
                //Read line
                line = scanner.nextLine();

                vector[i] = (Integer.parseInt(line));

                i++;
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return vector;
    }

    //Map labels to one hot variant
    private int[][] mapOneHotLabels(int[] labels, String type){
        int[][] oneHot;
        if(type != "test"){
            oneHot = new int [60000][10];
        }
        else{
            oneHot = new int [10000][10];
        }

        for(int i = 0; i < labels.length; i++){
            oneHot[i][labels[i]] = 1;
        }

        return oneHot;
    }

    //Normal easy normalization by dividing everything by the max value
    public void maxValueNormalization(){
        for(int i = 0; i < trainImages.length; i++){
            for(int j = 0; j < trainImages[i].length; j++){
                trainImages[i][j] = trainImages[i][j] / 255.0;
            }
        }
        for(int i = 0; i < testImages.length; i++){
            for(int j = 0; j < testImages[i].length; j++){
                testImages[i][j] = testImages[i][j] / 255.0;
            }
        }
    }
}
