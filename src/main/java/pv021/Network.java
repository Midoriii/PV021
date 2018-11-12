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

    private double[][] trainImages;
    private double[][] testImages;
    private int[] trainLabels;
    private int[] testLabels;
    private int[][] trainLabelsOneHot;
    private int[][] testLabelsOneHot;

    private int[] trainPredictions;
    private int[] testPredictions;


    //Init layers in constructor
    public Network(int inputDimensions, int trainImagesCount, int testImagesCount, int hiddenLayersCount, ArrayList<Integer> hiddenLayersNeuronCount){
        trainImages = new double[trainImagesCount][inputDimensions];
        testImages = new double[testImagesCount][inputDimensions];
        trainLabels = new int[trainImagesCount];
        testLabels = new int[testImagesCount];
        trainLabelsOneHot = new int[trainImagesCount][categories];
        testLabelsOneHot = new int[testImagesCount][categories];

        trainPredictions = new int[trainImagesCount];
        testPredictions = new int[testImagesCount];

        //Create desired layers
        inputLayer = new Layer(1, inputDimensions);
        for(int i = 0; i < hiddenLayersCount; i++){
            //First hidden layer has the weights number equal to the neurons in input layer
            if(i == 0){
                hiddenLayers.add(new Layer(inputDimensions, hiddenLayersNeuronCount.get(i)));
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
            vector = new double[trainImages.length][trainImages[0].length];
        }
        else{
            vector = new double[testImages.length][testImages[0].length];
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
            vector = new int[trainImages.length];
        }
        else{
            vector = new int[testImages.length];
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
            oneHot = new int [trainImages.length][categories];
        }
        else{
            oneHot = new int [testImages.length][categories];
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

    //Z-score normalization
    public void zScoreNormalization(){

        //Compute all the neccessary stuff for one image
        for(int i = 0; i < trainImages.length; i++){
            double average = 0.0;
            double sum = 0.0;
            double deviation = 0.0;

            //Get the sum for average
            for (int j = 0; j < trainImages[i].length; j++){
                sum += trainImages[i][j];
            }
            //Get the average
            average = sum / (double) trainImages[i].length;

            sum = 0.0;

            //Get the sum for deviation
            for (int j = 0; j < trainImages[i].length; j++){
                sum += Math.pow((trainImages[i][j] - average), 2.0);
            }

            //Get the deviation
            deviation = Math.sqrt((1.0/((double) trainImages[i].length - 1.0))* sum);

            //Compute the normalization
            for(int j = 0; j < trainImages[i].length; j++){
                trainImages[i][j] = (trainImages[i][j] - average) / deviation;
            }
        }

        //Compute all the neccessary stuff for one image
        for(int i = 0; i < testImages.length; i++){
            double average = 0.0;
            double sum = 0.0;
            double deviation = 0.0;

            //Get the sum for average
            for (int j = 0; j < testImages[i].length; j++){
                sum += testImages[i][j];
            }
            //Get the average
            average = sum / (double) testImages[i].length;

            sum = 0.0;

            //Get the sum for deviation
            for (int j = 0; j < testImages[i].length; j++){
                sum += Math.pow((testImages[i][j] - average), 2.0);
            }

            //Get the deviation
            deviation = Math.sqrt((1.0/((double) testImages[i].length - 1.0))* sum);

            //Compute the normalization
            for(int j = 0; j < testImages[i].length; j++){
                testImages[i][j] = (testImages[i][j] - average) / deviation;
            }
        }
    }
}
