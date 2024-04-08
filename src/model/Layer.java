package model;

import java.util.*;
public class Layer {

    public static final int INPUT_LAYER = 129;
    public static final int HIDDEN_LAYER = 130;
    public static final int OUTPUT_LAYER = 131;

    Layer input;
    Layer output;

    double[][] weights;
    double[] output_vector;

    double[] bias_weights;

    private int type;

    public Layer(int type) {
        this.type = type;
        switch (type) {
            case INPUT_LAYER : {
                output_vector = new double[NeuralNetwork.INPUT_DIGITS_SIZE];
                break;
            }
            case HIDDEN_LAYER : {
                output_vector = new double[NeuralNetwork.HIDDEN_DIGITS_SIZE];
                bias_weights = new double[output_vector.length];
                break;
            }
            default : {
                output_vector = new double[NeuralNetwork.OUTPUT_DIGITS_SIZE];
                bias_weights = new double[output_vector.length];
            }
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
        output_vector = Arrays.stream(image.phi()).mapToDouble(i -> i).toArray();
    }

    /**
     * "Fires" this layer of the neural network; in particular, calculates the output vector for this layer.
     *
     * Each layer must be fired in sequential order.
     *
     */
    public void fire() {
        if(this.type != INPUT_LAYER) {
            System.out.println("Firing layer.");
            System.out.println("Weights is " + weights.length + " by " + weights[0].length + " matrix");
            double[] input = this.input.output_vector;
            System.out.println("Incoming vector is " + input.length + " x 1");
            for(int i = 0; i < output_vector.length; i++) {
                output_vector[i] = dotproduct(input, weights[i]); //weights
                output_vector[i] += bias_weights[i]; //bias
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

    private static double sigmoid(double x) {
        double e = Math.exp(-x);
        return (1 / (1+e));
    }





}
