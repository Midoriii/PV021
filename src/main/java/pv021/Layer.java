package pv021;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;

@Getter
@Setter
public class Layer {

    private double[] inputs;
    private double[] outputs;
    private double[] localErrorGradients;
    private Neuron[] neurons;


    public Layer(int prevLayerNeuronsCount, int neuronCount){
        inputs = new double[prevLayerNeuronsCount];
        outputs = new double[neuronCount];
        localErrorGradients = new double[neuronCount];
        neurons = new Neuron[neuronCount];

        for(int i = 0; i < neuronCount; i++){
            //+1 for bias
            neurons[i] = new Neuron(prevLayerNeuronsCount + 1);
        }
        //Prep the outputs
        for(int i = 0; i < neuronCount; i++){
            outputs[i] = 0.0;
        }
    }

    //Calculate this layers' output by multiplying inputs with neuron weights
    public void calculateOutputs(String function){

        //For each neuron in this layer
        for(int i = 0; i < neurons.length; i++){
            //Let the neuron calculate its inner potential
            neurons[i].calculateInnerPotential(inputs);
        }

        for(int i = 0; i < neurons.length; i++){

            //If the neuron is in hidden layer, relu its inner potential
            if(function.equals("Relu")){
                //Set other outputs
                outputs[i] = neurons[i].relu();
            }
            else if(function.equals("Sigmoid")){
                outputs[i] = neurons[i].sigmoid();
            }
            //Else it is in output layer and we use softmax
            else{
                outputs[i] = neurons[i].softmax(neurons);
            }

        }
    }
}
