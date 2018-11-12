package pv021;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;

@Getter
@Setter
public class Layer {

    private double[] inputs;
    private double[] outputs;
    private ArrayList<Neuron> neurons = new ArrayList<Neuron>();


    public Layer(int prevLayerNeuronsCount, int neuronCount){
        inputs = new double[prevLayerNeuronsCount];
        outputs = new double[neuronCount];

        for(int i = 0; i < neuronCount; i++){
            neurons.add(new Neuron(prevLayerNeuronsCount));
        }
        //Prep the outputs
        for(int i = 0; i < neuronCount; i++){
            outputs[i] = 0.0;
        }
    }

    //Calculate this layers' output by multiplying inputs with neuron weights
    public void calculateOutputs(String function){

        //For each neuron in this layer
        for(int i = 0; i < neurons.size(); i++){
            //Let the neuron calculate its inner potential
            neurons.get(i).calculateInnerPotential(inputs);
        }

        for(int i = 0; i < neurons.size(); i++){

            //If the neuron is in hidden layer, relu its inner potential
            if(function.equals("Relu")){
                //Set other outputs
                outputs[i] = neurons.get(i).relu();
            }
            //Else it is in output layer and we use softmax
            else{
                outputs[i] = neurons.get(i).softmax(neurons);
            }

        }
    }
}
