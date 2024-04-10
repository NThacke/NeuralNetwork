package model;

import model.util.Util;

import java.util.*;
import java.io.*;
public class Driver implements Comparable<Driver> {

    public NeuralNetwork nn;

    public List<Image> images;

    List<Image> trainingset;

    public int[] labels;

    private int n;
    private int a;
    private int b;

    public double acc;

    private static final int TRAINING_CNT = 100;

    private double threshold;

    public Driver(int n, int a, int b, double threshold, List<Integer> hidden) {
        NeuralNetwork.INPUT_DIGITS_SIZE = n;
        images = new ArrayList<>();
        this.n = n;
        this.a = a;
        this.b = b;
        this.threshold = threshold;
        load_Neural_Net(hidden);
    }

    private void load_Neural_Net(List<Integer> hidden) {
        NeuralNetwork nn = loadNN("src/data/neural_nets/n:" + n + "_a:" + a + "_b:" + b + "_d:" + threshold);
        if(nn != null) {
            this.nn = nn;
        }
        else {
            this.nn = new NeuralNetwork(n, hidden, 10);
        }
    }

     // Method to save a Person object to a file
     private void saveNN(NeuralNetwork nn, String filename) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(nn);
            System.out.println("NN object saved successfully.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Method to load a Person object from a file
    private NeuralNetwork loadNN(String filename) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            return (NeuralNetwork) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public void testing() {
        loadImages(Util.DIGIT_VALIDATION_DATA);
        loadLabels(Util.DIGIT_VALIDATION_LABELS);
        Image image = images.get(0);
        System.out.println(image.toString());
        double[] output = nn.forward_propagation(image);
        System.out.println("Model gives");
        for(int i = 0; i < output.length; i++) {
            System.out.println(i + " : " + output[i]);
        }
        nn.back_propagate(image, output);
    }
    private double[] expected(Image image, double[] output_vector) {
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
                arr[i] = diff * diff;
            }
            return arr;
        }
        throw new IllegalArgumentException("Expected and output do not have the same length Expected : " + expected.length + " Output : " + output.length);
    }

    public void validate() {
        loadImages(Util.DIGIT_VALIDATION_DATA);
        loadLabels(Util.DIGIT_VALIDATION_LABELS);
        int correct = 0;
        for(int i = 0; i < images.size(); i++) {
            Image image = images.get(i);
            int answer = nn.fire(image);
            System.out.println(image);
            System.out.println(answer);
            if(answer == labels[image.getID()]) {
                correct++;
            }

        }
        System.out.println(images.size());
        double accuracy = (double)(correct)/(double)(images.size());
        this.acc = accuracy;
        System.out.println("Correct " + correct + " out of " + images.size() + " for an accuracy of " + accuracy);
    }

    public void train() {
        // loadImages(Util.DIGIT_TRAINING_DATA);
        // loadLabels(Util.DIGIT_TRAINING_LABELS);
        trainingSet(threshold);
        long cnt = 0;
        while(cnt < TRAINING_CNT) {
            if(cnt % 10 == 0) {
                System.out.println(cnt);
            }

            for (int i = 0; i < trainingset.size(); i++) {
                Image image = trainingset.get(i);
                nn.train(image);
            }
            cnt++;
        }
        saveNN(this.nn, "src/data/neural_nets/n:" + n + "_a:" + a + "_b:" + b + "_d:" + threshold);
    }

    public void randomizeWeights() {
        nn.randomizeWeights();
    }

    private void trainingSet(double d) {
        this.trainingset = new ArrayList<>();
        if(d == 1.0) {
            while(images.size() > 0) {
                int r = Util.random.nextInt(images.size());
                trainingset.add(images.remove(r));
            }
        }
        else {
            int cnt = (int)((d)*(images.size()));
            System.out.println(cnt);
            for(int i = 0; i < cnt; i++) {
                int r = Util.random.nextInt(images.size());
                trainingset.add(images.remove(r));
            }
        }
    }

    private void loadImages(String filename) {
        try {
            this.images = new ArrayList<>();
            RandomAccessFile file = new RandomAccessFile(filename, "r");
            int id = 0;
            while(file.getFilePointer() < file.length()) {
                Image image = new Image(n, a, b, file);
                image.setID(id);
                this.images.add(image);
                id++;
            }
            file.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

    private void loadLabels(String filename) {
        try {
            this.labels = new int[images.size()];
            Scanner scanner = new Scanner(new File(filename));
            int i = 0;
            while(scanner.hasNext()) {
                this.labels[i] = scanner.nextInt();
                images.get(i).setLabel(this.labels[i]);
                i++;
            }
            scanner.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

    public String toString() {
        return "N : " + n + " A : " + a + " B : " + b + " Threshold : " + threshold + " Accuracy : " + acc;
    }

    public int compareTo(Driver other) {
        return (int)(100*(this.acc - other.acc));
    }
}
