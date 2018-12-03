package pv021;

import lombok.Getter;
import lombok.Setter;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Scanner;

@Getter
@Setter
public class Network {

    //Hyper-parameters
    private double momentum = 0.9;
    private double decayRate = 0.0005;
    private double learningRate = 0.001;

    private int categories = 10; //2

    private Layer inputLayer;
    private ArrayList<Layer> hiddenLayers = new ArrayList<Layer>();
    private Layer outputLayer;

    private double[][] trainImages;
    private double[][] testImages;
    private int[] trainLabels;
    private int[] testLabels;
    private int[][] trainLabelsOneHot;
    //Are test really needed though
    private int[][] testLabelsOneHot;

    private int[] trainPredictions;
    private int[] testPredictions;

    double loss;


    //Init layers and arrays in constructor
    public Network(int inputDimensions, int trainImagesCount, int testImagesCount, int hiddenLayersCount, ArrayList<Integer> hiddenLayersNeuronCount){
        trainImages = new double[trainImagesCount][inputDimensions];
        testImages = new double[testImagesCount][inputDimensions];
        trainLabels = new int[trainImagesCount];
        testLabels = new int[testImagesCount];
        trainLabelsOneHot = new int[trainImagesCount][categories];
        testLabelsOneHot = new int[testImagesCount][categories];
        /*
        trainImages = new double[4][2];
        testImages = new double[4][2];
        trainLabels = new int[4];
        testLabels = new int[4];
        trainLabelsOneHot = new int[4][categories];
        testLabelsOneHot = new int[4][categories];*/

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

    //Training loop, might add batches TODO
    public void train(){
        for(int i = 0; i < trainImages.length; i++){
            //Feed input to the input layer
            getInput(i, "train");
            //Calculate forward pass
            forwardPass();
            //Calculate error
            calculateError(i);
            //Backpropagate the error
            backprop(i);
            //Update weights
            updateWeights();
            //Keep a record of predictions

            /*
            for(int l = 0; l < outputLayer.getOutputs().length; l++){
                System.out.println("Output " + l + ": " + outputLayer.getOutputs()[l]);
            }*/

            trainPredictions[i] = getPredictedLabel();
        }
    }

    //Prediction loop, implicitly predicts on the whole test set TODO
    public void predict(){
        for(int i = 0; i < testImages.length; i++){
            //Feed input to the input layer
            getInput(i, "test");
            //Calculate forward pass
            forwardPass();
            //Keep a record of predictions
            testPredictions[i] = getPredictedLabel();
        }
    }

    //Forward pass
    public void forwardPass(){

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

    //Cross-Entropy error function, without L2 regularization which is ensured by weight decay
    public void calculateError(int imageNumber){
        double sum = 0.0;
        for(int i = 0; i < outputLayer.getOutputs().length; i++){
            sum += Math.log(outputLayer.getOutputs()[i]) * trainLabelsOneHot[imageNumber][i];
        }
        loss = -1.0 * sum;
    }

    //Feed image array to the input layer
    private void getInput(int imageNumber, String test){
        //Inputs of the input layer are its own input
        if(test.equals("test")){
            inputLayer.setInputs(testImages[imageNumber]);
        }
        else{
            inputLayer.setInputs(trainImages[imageNumber]);
        }
        inputLayer.setOutputs(inputLayer.getInputs());
    }

    //Read train and test image vectors and labels //FIX
    public void readMnist(){
        String mnistTrainVectors = "mnist/mnist_train_vectors.csv";
        String mnistTestVectors = "mnist/mnist_test_vectors.csv";
        String mnistTrainLabels = "mnist/mnist_train_labels.csv";
        String mnistTestLabels = "mnist/mnist_test_labels.csv";
        /*
        trainImages[1][0] = 1.0;
        trainImages[1][1] = 1.0;

        trainImages[2][0] = 1.0;
        trainImages[2][1] = 0.0;

        trainImages[3][0] = 0.0;
        trainImages[3][1] = 0.0;

        trainImages[0][0] = 0.0;
        trainImages[0][1] = 1.0;

        testImages[1][0] = 1.0;
        testImages[1][1] = 1.0;

        testImages[2][0] = 1.0;
        testImages[2][1] = 0.0;

        testImages[3][0] = 0.0;
        testImages[3][1] = 0.0;

        testImages[0][0] = 0.0;
        testImages[0][1] = 1.0;

        trainLabels[0] = 1;
        trainLabels[1] = 0;
        trainLabels[2] = 1;
        trainLabels[3] = 0;

        testLabels[0] = 1;
        testLabels[1] = 0;
        testLabels[2] = 1;
        testLabels[3] = 0;

        */
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

    //Backpropagation algorithm TODO
    public void backprop(int imageNumber){
        //for each output neuron i
        for(int i = 0; i < outputLayer.getNeurons().size(); i++){
            //derivative = (softmax_i - true_one_hot_i) * softmax_i * (1 - softmax_i)
            outputLayer.getDerivatives()[i] = (outputLayer.getOutputs()[i] - (double) trainLabelsOneHot[imageNumber][i]) * outputLayer.getOutputs()[i] * (1.0 - outputLayer.getOutputs()[i]);
            //for each his weight n (for weight_0 input is 1.0)
            for(int n = 0; n < outputLayer.getNeurons().get(i).getWeights().length; n++){
                //delta weight = derivative * input_n
                if(n == 0){
                    outputLayer.getNeurons().get(i).getDeltaWeights()[n] = outputLayer.getDerivatives()[i] * 1.0;
                }
                else{
                    outputLayer.getNeurons().get(i).getDeltaWeights()[n] = outputLayer.getDerivatives()[i] * outputLayer.getInputs()[n-1];
                }
            }
        }

        //for each hidden layer (from the top)
        for(int l = hiddenLayers.size() - 1; l >= 0; l--){
            //for each hidden neuron i
            for(int i = 0; i < hiddenLayers.get(l).getNeurons().size(); i++){
                double derivatives_sum = 0.0;
                //for each neuron j up a layer
                if(l == hiddenLayers.size() - 1){
                    //prev_derivatives_sum += prev_layer_derivative * j's_weight_i+1
                    for(int j = 0; j < outputLayer.getNeurons().size(); j++){
                        derivatives_sum += (outputLayer.getDerivatives()[j] * outputLayer.getNeurons().get(j).getWeights()[i+1]);
                    }
                }
                else{
                    //prev_derivatives_sum += prev_layer_derivative * j's_weight_i+1
                    for(int j = 0; j < hiddenLayers.get(l + 1).getNeurons().size(); j++){
                        derivatives_sum += (hiddenLayers.get(l + 1).getDerivatives()[j] * hiddenLayers.get(l + 1).getNeurons().get(j).getWeights()[i+1]);
                    }
                }
                //derivative = prev_derivatives_sum * act_function_derivative
                hiddenLayers.get(l).getDerivatives()[i] = derivatives_sum * hiddenLayers.get(l).getNeurons().get(i).reluPrime();
                //for each his weight n
                for(int n = 0; n < hiddenLayers.get(l).getNeurons().get(i).getWeights().length; n++){
                    //delta weight = derivative * input_n * learning_rate
                    if(n == 0){
                        hiddenLayers.get(l).getNeurons().get(i).getDeltaWeights()[n] = hiddenLayers.get(l).getDerivatives()[i] * learningRate * 1.0;
                    }
                    else{
                        hiddenLayers.get(l).getNeurons().get(i).getDeltaWeights()[n] = hiddenLayers.get(l).getDerivatives()[i] * learningRate * hiddenLayers.get(l).getInputs()[n-1];
                    }
                }
            }
        }
    }

    //Update weights, along with momentum and weight decay TODO
    public void updateWeights(){
        //for each output layer neuron
        for(int i = 0; i < outputLayer.getNeurons().size(); i++){
            //for each weight
            for(int j = 0; j < outputLayer.getNeurons().get(i).getWeights().length; j++){
                //weight += delta_weight + momentum * prev_delta_weight
                outputLayer.getNeurons().get(i).getWeights()[j] += outputLayer.getNeurons().get(i).getDeltaWeights()[j] + (momentum * outputLayer.getNeurons().get(i).getPrevDeltaWeights()[j]);
                //prev_delta_weight = delta_weight;
                outputLayer.getNeurons().get(i).getPrevDeltaWeights()[j] = outputLayer.getNeurons().get(i).getDeltaWeights()[j];
            }

        }
        //the same for each hidden layer
        for(int l = 0; l < hiddenLayers.size(); l++){
            for(int i = 0; i < hiddenLayers.get(l).getNeurons().size(); i++){
                for(int j = 0; j < hiddenLayers.get(l).getNeurons().get(i).getWeights().length; j++){
                    hiddenLayers.get(l).getNeurons().get(i).getWeights()[j] += hiddenLayers.get(l).getNeurons().get(i).getDeltaWeights()[j] + (momentum * hiddenLayers.get(l).getNeurons().get(i).getPrevDeltaWeights()[j]);
                    hiddenLayers.get(l).getNeurons().get(i).getPrevDeltaWeights()[j] = hiddenLayers.get(l).getNeurons().get(i).getDeltaWeights()[j];
                }
            }
        }
    }

    //Find the label the net predicts for the current input (index = label in this case)
    public int getPredictedLabel(){

        int predictionIndex = 0;
        double max = 0.0;

        //Search for the maximum probability and get its index = label
        for(int j = 0; j < outputLayer.getOutputs().length; j++){
            if(max < outputLayer.getOutputs()[j]){
                max = outputLayer.getOutputs()[j];
                predictionIndex = j;
            }
        }
        return predictionIndex;
    }

    //Print predictions as required TODO
    public void printPredictions(){
        try {
            PrintWriter testPredictionswriter = new PrintWriter("actualTestPredictions", "UTF-8");
            PrintWriter trainPredictionswriter = new PrintWriter("trainPredictions", "UTF-8");
            PrintWriter expectedPredictionswriter = new PrintWriter("expectedPredictions", "UTF-8");

            for(int i = 0; i < trainPredictions.length; i++){
                trainPredictionswriter.println(trainPredictions[i]);
            }
            for(int i = 0; i < testPredictions.length; i++){
                testPredictionswriter.println(testPredictions[i]);
            }
            for(int i = 0; i < testLabels.length; i++){
                expectedPredictionswriter.println(testLabels[i]);
            }

            testPredictionswriter.close();
            trainPredictionswriter.close();
            expectedPredictionswriter.close();

        } catch (Exception e) {

        }
    }

    //Print the correctness percentage
    public void printPercentages(){
        double trainCorrect = 0.0;
        double testCorrect = 0.0;

        for(int i = 0; i < trainPredictions.length; i++){
            if(trainPredictions[i] == trainLabels[i]){
                trainCorrect++;
            }
        }
        for(int i = 0; i < testPredictions.length; i++){
            if(testPredictions[i] == testLabels[i]){
                testCorrect++;
            }
        }
        System.out.println("-----------------------------------------------------------");
        System.out.println("Training accuracy: " + trainCorrect/trainPredictions.length * 100.0 + "%.");
        System.out.println("Test accuracy: " + testCorrect/testPredictions.length * 100.0 + "%.");
        System.out.println("-----------------------------------------------------------");
    }
}
