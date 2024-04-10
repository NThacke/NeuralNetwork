package model;

import java.io.Serializable;
import java.util.*;

public class NeuralNetwork implements Serializable {

    public static int INPUT_DIGITS_SIZE = 196;
    public static int OUTPUT_DIGITS_SIZE = 10;

    private static final long serialVersionUID = 6529685098267757690L;

    List<Layer> hiddenLayers;

    Layer input_layer;
    Layer output_layer;

    int input;

    int hidden;

    int output;

    /**
     * A three layer neural network.
     * @param input The number of input nodes
     * @param hiddenNodes A List containing the number of nodes in each hidden layer of this Neural Network.
     * @param output The number of output nodes
     */
    public NeuralNetwork(int input, List<Integer> hiddenNodes, int output) {
        INPUT_DIGITS_SIZE = input;
        input_layer = new Layer(Layer.INPUT_LAYER, input);
        this.input = input;
        this.output = output;
        hiddenLayers = new ArrayList<>();
        for(int i = 0; i< hiddenNodes.size(); i++) {
            Layer layer = new Layer(Layer.HIDDEN_LAYER, hiddenNodes.get(i));
            layer.id = i+1;
            hiddenLayers.add(layer);
            if(i == 0) {
                input_layer.output = layer;
                layer.input = input_layer;
            }
            else {
                layer.input = hiddenLayers.get(i-1);
                hiddenLayers.get(i-1).output = layer;
            }
            layer.init_connections();
        }

        output_layer = new Layer(Layer.OUTPUT_LAYER, output);
        output_layer.input = hiddenLayers.get(hiddenLayers.size()-1);
        hiddenLayers.get(hiddenLayers.size()-1).output = output_layer;
        output_layer.init_connections();
    }

    public void train(Image image) {
        double[] arr = forward_propagation(image);
        back_propagate(image, arr);
    }

    public double[] forward_propagation(Image image) {
        input_layer.setImage(image);
        for(Layer layer : hiddenLayers) {
            layer.fire();
        }
        output_layer.fire();
        return output_layer.output_vector;
    }

    public int fire(Image image) {
        double[] arr = forward_propagation(image);
        int max = 0;
        for(int i = 0; i< arr.length; i++) {
            if(arr[i] > arr[max]) {
                max = i;
            }
        }
        return max;
    }

    public void back_propagate(Image image, double[] output) {
        output_layer.back_propagate(image, output);
    }

    public void randomizeWeights() {
        for(Layer layer : hiddenLayers) {
            layer.randomizeWeights();
        }
        output_layer.randomizeWeights();
    }
}
