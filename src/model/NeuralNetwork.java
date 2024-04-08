package model;

import java.util.*;

public class NeuralNetwork {

    public static final int INPUT_DIGITS_SIZE = 784;
    public static final int HIDDEN_DIGITS_SIZE = 329;
    public static final int OUTPUT_DIGITS_SIZE = 10;

    Layer input_layer;
    Layer hidden_layer;
    Layer output_layer;

    int input;

    int hidden;

    int output;

    /**
     * A three layer neural network.
     * @param input The number of input nodes
     * @param hidden The number of hidden nodes
     * @param output The number of output nodes
     */
    public NeuralNetwork(int input, int hidden, int output) {
        input_layer = new Layer(Layer.INPUT_LAYER);
        hidden_layer = new Layer(Layer.HIDDEN_LAYER);
        output_layer = new Layer(Layer.OUTPUT_LAYER);

        input_layer.output = hidden_layer;
        
        hidden_layer.input = input_layer;
        hidden_layer.output = output_layer;

        output_layer.input = hidden_layer;
        
        hidden_layer.init_connections();
        output_layer.init_connections();

        this.input = input;
        this.hidden = hidden;
        this.output = output;
    }

    public double[] forward_propogation(Image image) {
        input_layer.setImage(image);
        hidden_layer.fire();
        output_layer.fire();
        return output_layer.output_vector;
    }
}
