package model;

import model.util.Util;

import java.io.Serializable;

public class Layer implements Serializable {

    public static final double alpha = 0.01;

    public static final int INPUT_LAYER = 129;
    public static final int HIDDEN_LAYER = 130;
    public static final int OUTPUT_LAYER = 131;

    private static final long serialVersionUID = 623923109829327690L;

    Layer input;
    Layer output;

    double[][] weights;
    double[] output_vector;

    double[] bias_weights;

    double[] z;

    private int type;

    int id;

    private int size;

    public Layer(int type, int size) {
        this.type = type;
        this.size = size;
        switch (type) {
            case INPUT_LAYER : {
                output_vector = new double[size];
                z = new double[output_vector.length];
                break;
            }
            case HIDDEN_LAYER : {
                output_vector = new double[size];
                z = new double[output_vector.length];
                bias_weights = new double[output_vector.length];
                break;
            }
            default : {
                output_vector = new double[size];
                z = new double[output_vector.length];
                bias_weights = new double[output_vector.length];
            }
        }
    }

    public void randomizeWeights() {
        System.out.println(id);
        double r = Math.sqrt(6.0/ (double)(input.output_vector.length));
        for(int i = 0; i < weights.length; i++) {
            for(int j = 0; j < weights[i].length; j++) {
                weights[i][j] = Util.random.nextGaussian();
            }
        }
        for(int i = 0; i < bias_weights.length; i++) {
            bias_weights[i] = Util.random.nextGaussian();
        }
    }

    public void init_connections() {
        switch (type) {
            case HIDDEN_LAYER : {
                weights = new double[output_vector.length][this.input.output_vector.length];
                break;
            }
            case OUTPUT_LAYER : {
                weights = new double[output_vector.length][this.input.output_vector.length];
                break;
            }
        }
    }

    public void setImage(Image image) {
        output_vector = image.phi();
    }

    /**
     * "Fires" this layer of the neural network; in particular, calculates the output vector for this layer.
     *
     * Each layer must be fired in sequential order.
     *
     */
    public void fire() {
        if(this.type != INPUT_LAYER) {
//            System.out.println("Firing layer.");
//            System.out.println("Weights is " + weights.length + " by " + weights[0].length + " matrix");
            double[] input = this.input.output_vector;
//            System.out.println("Incoming vector is " + input.length + " x 1");
            for(int i = 0; i < output_vector.length; i++) {
                output_vector[i] = dotproduct(input, weights[i]); //weights
                output_vector[i] += bias_weights[i]; //bias
                z[i] = output_vector[i];
                output_vector[i] = sigmoid(output_vector[i]);   
            }
        }
    }

    private double dotproduct(double[] arr1, double[] arr2) {
        if(arr1.length == arr2.length) {
            double dot = 0.0;
            for(int i = 0; i < arr1.length; i++) {
                dot += (arr1[i] * arr2[i]);
            }
            return dot;
        }
        throw new IllegalArgumentException("Cannot perform dot product of two vectors of unequal length");
    }

    /**
     * Propagates backwards on *this* layer, denoted layer L.
     *
     *     *   -   *   -   *
     *
     *     *   -   *   -   *
     *
     *     *   -   *   -   *
     *
     *   L - 1     L     L + 1
     *
     * @param error The error terms for each neuron in the layer L + 1.
     */
    protected void back_propagate(double[] error) {
        if(type != INPUT_LAYER) {
            double[] e = new double[this.weights.length]; //the error that neuron i has
            double[][] weights_subsequent = output.weights;

            for (int i = 0; i < this.weights.length; i++) {
                double error_i = 0.0; //the error for the ith neuron in layer L
               double sig = sigmoid_prime(z[i]);
                for (int j = 0; j < error.length; j++) {
                    error_i += (error[j] * weights_subsequent[j][i]);
                }
//                e[i] = error_i * sig;
                e[i] = (error_i * sig);
            }

//            System.out.println("The error terms are");
//            for(int i = 0; i < e.length; i++) {
//                System.out.println(e[i]);
//            }

            //Before updating, we need to propagate backwards.
            input.back_propagate(e);

            //Now that we have our errors for each neuron, we can calculate the change in each weight.
            // Δ w_{j,k} ^{L} = - alpha * error[k] ^ {L} * activation[k] ^ {L - 1}

            double[] activation = input.output_vector;
//            System.out.println("Error has length " + error.length);
//            System.out.println("Activation has length " + activation.length);
//            System.out.println("Weights is " + weights.length + " x " + weights[0].length);

            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] -= ((alpha) * (e[i]) * activation[j]);
                }
            }

            //update bias
            for(int i = 0; i < bias_weights.length; i++) {
                bias_weights[i] -= (alpha * e[i]);
            }

        }
    }

    protected double sigmoid_prime(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    // Sigmoid activation function
    private double sigmoid(double x) {
        if (x >= 0) {
            return 1.0 / (1.0 + Math.exp(-x));
        } else {
            double expX = Math.exp(x);
            return expX / (1.0 + expX);
        }
    }
    public void back_propagate(Image image, double[] output) {

//        if(type == HIDDEN_LAYER) {
//            System.out.println("Hidden layer");
//        }
//        else if(type == OUTPUT_LAYER) {
//            System.out.println("Output Layer");
//        }
        double[] expected = expected(image);
        double[] loss = loss(expected, output);

//        System.out.println("The error is :");
//        for(int i = 0; i<loss.length; i++) {
//            System.out.println(loss[i]);
//        }

        double[] neuron_error = new double[loss.length]; //the error that each neuron contributes at the output layer
        for(int i = 0; i < neuron_error.length; i++) {
            neuron_error[i] = loss[i] * sigmoid_prime(z[i]);
        }

//        System.out.println("The error term for each neuron is :");
//        for(int i = 0; i < neuron_error.length; i++) {
//            System.out.println(neuron_error[i]);
//        }
//        System.out.println("Z[i] | sigmoid'[i] : ");
//        for(int i = 0; i<z.length; i++) {
//            System.out.println(z[i] + " | " + sigmoid_prime(z[i]));
//        }
        input.back_propagate(neuron_error);

        //weights
        double[] activation = input.output_vector;
        for(int i = 0; i < weights.length; i++) {
            for(int j = 0; j < weights[i].length; j++) {
//                System.out.println("Updating weight by " + (-alpha * neuron_error[i] * activation[j]));
                weights[i][j] -= (alpha * neuron_error[i] * activation[j]);
            }
        }
        //biases
        for(int i = 0; i < bias_weights.length; i++) {
            bias_weights[i] -= (alpha)*(neuron_error[i]);
        }
    }


    private double[] expected(Image image) {
        double[] arr =  new double[output_vector.length];
        int ans = image.getLabel();
        for(int i = 0; i < arr.length; i++) {
            if(i == ans) {
                arr[i] = 1.0;
            }
            else {
                arr[i] = 0.0;
            }
        }
        return arr;
    }

    private double[] loss(double[] expected, double[] output) {
        if(expected.length == output.length) {
            double[] arr = new double[expected.length];
            for(int i = 0; i < output.length; i++) {
                double diff = output[i] - expected[i];
                arr[i] = diff;
            }
            return arr;
        }
        throw new IllegalArgumentException("Expected and output do not have the same length Expected : " + expected.length + " Output : " + output.length);
    }
}
