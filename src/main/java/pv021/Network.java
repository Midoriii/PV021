package pv021;

import lombok.Getter;
import lombok.Setter;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

@Getter
@Setter
public class Network {

    //Hyper-parameters
    private double momentum = 0.9;
    private double decayRate = 0.0005;
    private double learningRate = 0.001;

    private int categories = 10;

    private Layer inputLayer;
    private Layer[] hiddenLayers;
    private Layer outputLayer;

    private double[][] trainImages;
    private double[][] validateImages;
    private double[][] testImages;
    private int[] trainLabels;
    private int[] validateLabels;
    private int[] testLabels;
    private int[][] trainLabelsOneHot;
    private int[][] validateLabelsOneHot;
    //Are test really needed though
    private int[][] testLabelsOneHot;

    private int[] trainPredictions;
    private int[] validatePredictions;
    private int[] testPredictions;

    double validationAccuracy = 0.0;

    double loss;
    double loss_sum;


    //Init layers and arrays in constructor
    public Network(int inputDimensions, int trainImagesCount, int validateImagesCount, int testImagesCount, int hiddenLayersCount, ArrayList<Integer> hiddenLayersNeuronCount){
        trainImages = new double[trainImagesCount][inputDimensions];
        validateImages = new double[validateImagesCount][inputDimensions];
        testImages = new double[testImagesCount][inputDimensions];
        trainLabels = new int[trainImagesCount];
        validateLabels = new int[validateImagesCount];
        testLabels = new int[testImagesCount];
        trainLabelsOneHot = new int[trainImagesCount][categories];
        validateLabelsOneHot = new int[validateImagesCount][categories];
        testLabelsOneHot = new int[testImagesCount][categories];

        trainPredictions = new int[trainImagesCount];
        validatePredictions = new int[validateImagesCount];
        testPredictions = new int[testImagesCount];

        hiddenLayers = new Layer[hiddenLayersCount];

        //Create desired layers
        inputLayer = new Layer(1, inputDimensions);
        for(int i = 0; i < hiddenLayersCount; i++){
            //First hidden layer has the weights number equal to the neurons in input layer
            if(i == 0){
                hiddenLayers[i] = new Layer(inputDimensions, hiddenLayersNeuronCount.get(i));
            }
            else{
                hiddenLayers[i] = new Layer(hiddenLayersNeuronCount.get(i-1), hiddenLayersNeuronCount.get(i));
            }
            //Output layer has the number of weights equal to the number of neurons in the last hidden layer
        outputLayer = new Layer(hiddenLayersNeuronCount.get(hiddenLayersNeuronCount.size() - 1), categories);
        }
    }

    //Training loop, might add batches TODO
    public void train(){
        loss_sum = 0.0;
        for(int i = 0; i < trainImages.length; i++){
            //Feed input to the input layer
            getInput(i, "train");
            //Calculate forward pass
            forwardPass();
            //Calculate error
            calculateError(i);

            loss_sum += loss;

            //Backpropagate the error
            backprop(i);
            //Update weights
            updateWeights();
            //Keep a record of predictions

            /*
            if(i == 0){
                for(int l = 0; l < outputLayer.getOutputs().length; l++){
                    System.out.println("Output " + l + ": " + outputLayer.getOutputs()[l]);
                }
            }*/
            trainPredictions[i] = getPredictedLabel();
        }
        System.out.println("loss: " + loss_sum / trainImages.length);
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
        double testCorrect = 0.0;
        for(int i = 0; i < testPredictions.length; i++){
            if(testPredictions[i] == testLabels[i]){
                testCorrect++;
            }
        }
        System.out.println("-----------------------------------------------------------");
        System.out.println("Test accuracy: " + testCorrect/testPredictions.length * 100.0 + "%.");
        System.out.println("-----------------------------------------------------------");
    }

    //Prediction loop, implicitly predicts on the whole validation set TODO
    public void validation(){
        for(int i = 0; i < validateImages.length; i++){
            //Feed input to the input layer
            getInput(i, "validation");
            //Calculate forward pass
            forwardPass();
            //Keep a record of predictions
            validatePredictions[i] = getPredictedLabel();
        }
    }

    //Forward pass
    public void forwardPass(){

        //Pass the previous outputs into current inputs, calculate activation, pass the outputs
        for(int i = 0; i < hiddenLayers.length; i++){

            //Set the inputs to the previous layer's outputs
            if(i == 0){
                hiddenLayers[i].setInputs(inputLayer.getOutputs());
            }
            else{
                hiddenLayers[i].setInputs(hiddenLayers[i-1].getOutputs());
            }

            //Calculate new output
            hiddenLayers[i].calculateOutputs("Relu");
        }

        //Calculate output Layer outputs
        outputLayer.setInputs(hiddenLayers[hiddenLayers.length-1].getOutputs());
        outputLayer.calculateOutputs("Sigmoid");
    }

    //Cross-Entropy error function, without L2 regularization which is ensured by weight decay
    public void calculateError(int imageNumber){
        /*double sum = 0.0;
        for(int i = 0; i < outputLayer.getOutputs().length; i++){
            sum += Math.log(outputLayer.getOutputs()[i]) * trainLabelsOneHot[imageNumber][i];
        }
        loss = -1.0 * sum;*/

        //MSE
        double sum = 0.0;
        for(int i = 0; i < outputLayer.getOutputs().length; i++){
            sum += Math.pow(trainLabelsOneHot[imageNumber][i] - outputLayer.getOutputs()[i], 2);
        }
        loss = sum/2.0;
    }

    //Feed image array to the input layer
    private void getInput(int imageNumber, String test){
        //Inputs of the input layer are its own input
        if(test.equals("test")){
            //inputLayer.setInputs(testImages[imageNumber]);
            inputLayer.setOutputs(testImages[imageNumber]);
        }
        else if(test.equals("train")){
            //inputLayer.setInputs(trainImages[imageNumber]);
            inputLayer.setOutputs(trainImages[imageNumber]);
        }
        else if(test.equals("validation")){
            inputLayer.setOutputs(validateImages[imageNumber]);
        }
    }

    //Read train and test image vectors and labels //FIX
    public void readMnist(){
        String mnistTrainVectors = "mnist/mnist_train_vectors.csv";
        String mnistTestVectors = "mnist/mnist_test_vectors.csv";
        String mnistTrainLabels = "mnist/mnist_train_labels.csv";
        String mnistTestLabels = "mnist/mnist_test_labels.csv";

        trainImages = parseVectors(mnistTrainVectors, "train");
        validateImages = parseVectors(mnistTrainVectors, "validate");
        testImages = parseVectors(mnistTestVectors, "test");
        trainLabels = parseLabels(mnistTrainLabels, "train");
        validateLabels = parseLabels(mnistTrainLabels, "validate");
        testLabels = parseLabels(mnistTestLabels, "test");

        trainLabelsOneHot = mapOneHotLabels(trainLabels, "train");
        validateLabelsOneHot = mapOneHotLabels(validateLabels, "validate");
        testLabelsOneHot = mapOneHotLabels(testLabels, "test");
    }

    //Parse those .csv files
    private double[][] parseVectors(String path, String type){
        String line;
        String cvsSplitBy = ",";
        double[][] vector;
        if(type == "test"){
            vector = new double[testImages.length][testImages[0].length];
        }
        else if(type == "validate"){
            vector = new double[validateImages.length][validateImages[0].length];
        }
        else{
            vector = new double[trainImages.length][trainImages[0].length];
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

                if(j >= trainImages.length && type == "validate"){
                    //Parse strings into doubles and add the single image into the set
                    for(int i = 0; i < addition.length; i++){
                        vector[j-trainImages.length][i] = (Double.parseDouble(addition[i]));
                    }
                }
                if(j < trainImages.length && type == "train"){
                    //Parse strings into doubles and add the single image into the set
                    for(int i = 0; i < addition.length; i++){
                        vector[j][i] = (Double.parseDouble(addition[i]));
                    }
                }
                else if(type == "test"){
                    //Parse strings into doubles and add the single image into the set
                    for(int i = 0; i < addition.length; i++){
                        vector[j][i] = (Double.parseDouble(addition[i]));
                    }
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
        if(type == "test"){
            vector = new int[testImages.length];
        }
        else if(type == "validate"){
            vector = new int[validateImages.length];
        }
        else{
            vector = new int[trainImages.length];
        }

        //Get file from resources folder
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource(path).getFile());

        try (Scanner scanner = new Scanner(file)) {
            int i = 0;
            while (scanner.hasNextLine()) {
                //Read line
                line = scanner.nextLine();

                if(type == "test"){
                    vector[i] = (Integer.parseInt(line));
                }
                else if(type == "validate" && i >= trainImages.length){
                    vector[i-trainImages.length] = (Integer.parseInt(line));
                }
                else if(type == "train" && i < trainImages.length){
                    vector[i] = (Integer.parseInt(line));
                }

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
        if(type == "test"){
            oneHot = new int[testImages.length][categories];
        }
        else if(type == "validate"){
            oneHot = new int[validateImages.length][categories];
        }
        else{
            oneHot = new int[trainImages.length][categories];
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

    //Backpropagation algorithm TODO: make it work
    public void backprop(int imageNumber){
        //for each output neuron i
        for(int i = 0; i < outputLayer.getNeurons().length; i++){
            //error_gradient = (softmax_i - true_one_hot_i) * softmax_i * (1 - softmax_i)
            outputLayer.getLocalErrorGradients()[i] = ((double) trainLabelsOneHot[imageNumber][i] - outputLayer.getOutputs()[i]) * outputLayer.getOutputs()[i] * (1.0 - outputLayer.getOutputs()[i]);
            //for each his weight n (for weight_0 input is 1.0)
            for(int n = 0; n < outputLayer.getNeurons()[i].getWeights().length; n++){
                //delta weight = learning_rate * error_gradient * input_n
                if(n == 0){
                    outputLayer.getNeurons()[i].getDeltaWeights()[n] = learningRate * outputLayer.getLocalErrorGradients()[i] * 1.0;
                }
                else{
                    outputLayer.getNeurons()[i].getDeltaWeights()[n] = learningRate * outputLayer.getLocalErrorGradients()[i] * outputLayer.getInputs()[n-1];
                }
            }
        }

        //for each hidden layer (from the top)
        for(int l = hiddenLayers.length - 1; l >= 0; l--){
            //for each hidden neuron i
            for(int i = 0; i < hiddenLayers[l].getNeurons().length; i++){
                double localErrorGradientsSum = 0.0;
                //for each neuron j up a layer
                if(l == hiddenLayers.length - 1){
                    //prev_error_gradients_sum += prev_error_gradient * j's_weight_i+1
                    for(int j = 0; j < outputLayer.getNeurons().length; j++){
                        localErrorGradientsSum += (outputLayer.getLocalErrorGradients()[j] * outputLayer.getNeurons()[j].getWeights()[i+1]);
                    }
                }
                else{
                    //prev_error_gradients_sum += prev_error_gradient * j's_weight_i+1
                    for(int j = 0; j < hiddenLayers[l + 1].getNeurons().length; j++){
                        localErrorGradientsSum += (hiddenLayers[l + 1].getLocalErrorGradients()[j] * hiddenLayers[l + 1].getNeurons()[j].getWeights()[i+1]);
                    }
                }
                //error_gradient = prev_error_gradients_sum * act_function_derivative
                hiddenLayers[l].getLocalErrorGradients()[i] = localErrorGradientsSum * hiddenLayers[l].getNeurons()[i].reluPrime();
                //for each his weight n
                for(int n = 0; n < hiddenLayers[l].getNeurons()[i].getWeights().length; n++){
                    //delta weight = error_gradient * input_n * learning_rate
                    if(n == 0){
                        hiddenLayers[l].getNeurons()[i].getDeltaWeights()[n] = learningRate * hiddenLayers[l].getLocalErrorGradients()[i] * 1.0;
                    }
                    else{
                        hiddenLayers[l].getNeurons()[i].getDeltaWeights()[n] = learningRate * hiddenLayers[l].getLocalErrorGradients()[i] * hiddenLayers[l].getInputs()[n-1];
                    }
                }
            }
        }
    }

    //Update weights, along with momentum and weight decay TODO: weight decay
    public void updateWeights(){
        //for each output layer neuron
        for(int i = 0; i < outputLayer.getNeurons().length; i++){
            //for each weight
            for(int j = 0; j < outputLayer.getNeurons()[i].getWeights().length; j++){
                //new delta with momentum
                outputLayer.getNeurons()[i].getDeltaWeights()[j] += momentum * outputLayer.getNeurons()[i].getPrevDeltaWeights()[j];
                //weight += delta_weight + momentum * prev_delta_weight
                outputLayer.getNeurons()[i].getWeights()[j] += outputLayer.getNeurons()[i].getDeltaWeights()[j];
                //prev_delta_weight = delta_weight;
                outputLayer.getNeurons()[i].getPrevDeltaWeights()[j] = outputLayer.getNeurons()[i].getDeltaWeights()[j];
            }

        }
        //the same for each hidden layer
        for(int l = 0; l < hiddenLayers.length; l++){
            for(int i = 0; i < hiddenLayers[l].getNeurons().length; i++){
                for(int j = 0; j < hiddenLayers[l].getNeurons()[i].getWeights().length; j++){
                    hiddenLayers[l].getNeurons()[i].getDeltaWeights()[j] += momentum * hiddenLayers[l].getNeurons()[i].getPrevDeltaWeights()[j];
                    hiddenLayers[l].getNeurons()[i].getWeights()[j] += hiddenLayers[l].getNeurons()[i].getDeltaWeights()[j];
                    hiddenLayers[l].getNeurons()[i].getPrevDeltaWeights()[j] = hiddenLayers[l].getNeurons()[i].getDeltaWeights()[j];
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

    //Print predictions as required TODO: expected too ?
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
        double validateCorrect = 0.0;

        for(int i = 0; i < trainPredictions.length; i++){
            if(trainPredictions[i] == trainLabels[i]){
                trainCorrect++;
            }
        }
        for(int i = 0; i < validatePredictions.length; i++){
            if(validatePredictions[i] == validateLabels[i]){
                validateCorrect++;
            }
        }
        validationAccuracy = validateCorrect/validatePredictions.length * 100.0;
        System.out.println("-----------------------------------------------------------");
        System.out.println("Training accuracy: " + trainCorrect/trainPredictions.length * 100.0 + "%.");
        System.out.println("Validation accuracy: " + validationAccuracy + "%.");
        System.out.println("-----------------------------------------------------------");
    }
}
