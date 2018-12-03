package pv021;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

@Getter
@Setter
public class Neuron {

    private double[] weights;
    private double[] deltaWeights;
    private double[] prevDeltaWeights;

    private double output;
    private double innerPotential;
    private double leakyReluCoef = 20.0;


    //Create proper number of weights, +1 is the bias, every input will have index 0 of value 1.0
    public Neuron(int numberOfWeights){
        weights = new double[numberOfWeights + 1];
        deltaWeights = new double[numberOfWeights + 1];
        prevDeltaWeights = new double[numberOfWeights + 1];

        for(int i = 0; i < numberOfWeights + 1; i++){
            weights[i] = addWeight(numberOfWeights);
        }
    }

    //Get the inner potential of this neuron, based on inputs * weights
    public void calculateInnerPotential(double[] inputs){
        double sum = 0.0;
        for(int i = 0; i < inputs.length; i++){
            //Input for bias
            if(i == 0){
                sum += 1.0 * weights[i];
            }
            else{
                sum += inputs[i] * weights[i];
            }
        }
        innerPotential = sum;
    }

    //Leaky Relu activation function
    public double relu(){
        if(innerPotential > 0){
            output = innerPotential;
            return output;
        }
        else{
            output = innerPotential / leakyReluCoef;
            return output;
        }
    }

    //Derivation of leaky Relu
    public double reluPrime(){
        if(innerPotential > 0){
            return 1.0;
        }
        else{
            return 1.0 / leakyReluCoef;
        }
    }

    //Softmax activation
    public double softmax(ArrayList<Neuron> outputNeurons){
        //Calculate the sum of e^(inner potential) of all output neurons
        double sum = 0.0;

        //To stabilise the softmax, we need to find and subtract the max value
        ArrayList<Double> innerPotentials = new ArrayList<Double>();
        for(int i = 0; i < outputNeurons.size(); i++){
            innerPotentials.add(outputNeurons.get(i).getInnerPotential());
        }

        //The maximum
        double max = Collections.max(innerPotentials);

        for(int i = 0; i < outputNeurons.size(); i++){
            sum += Math.exp(outputNeurons.get(i).getInnerPotential() - max);
        }
        //Return the softmax and set the output of this neuron
        output = Math.exp(innerPotential - max) / sum;
        return output;
    }

    //Appropriate weight init for Relu
    private double addWeight(int numberOfWeights){
        Random rand = new Random();
        return rand.nextGaussian() * Math.sqrt(2.0/numberOfWeights);
    }
}
