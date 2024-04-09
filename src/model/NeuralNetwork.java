package model;

public class NeuralNetwork {

    public static int INPUT_DIGITS_SIZE = 196;
    public static final int HIDDEN_DIGITS_SIZE = 32;
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

    public void train(Image image) {
        double[] arr = forward_propagation(image);
        back_propagate(image, arr);
//        output_layer.backpropagate(arr);
    }

    public double[] forward_propagation(Image image) {
        input_layer.setImage(image);
        hidden_layer.fire();
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

    void save(int n, int a, int b, double d) {
        hidden_layer.save(n, a, b, d);
        output_layer.save(n, a, b, d);
    }

    public void load(int n, int a, int b, double d) {
        output_layer.load(n, a, b, d);
        hidden_layer.load(n, a, b, d);
    }

    public void randomizeWeights() {
        hidden_layer.randomizeWeights();
        output_layer.randomizeWeights();
    }
}
