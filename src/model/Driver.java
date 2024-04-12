package model;

import model.util.Util;

import java.util.*;
import java.io.*;
public class Driver implements Comparable<Driver>, Util {

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

    private int type;

    public static final String NN_DIGITS_DIR = "src/data/neural_nets/digits/";

    public static final String NN_FACES_DIR = "src/data/neural_nets/faces/";

    private long time;

    public Driver(int n, int a, int b, double threshold, List<Integer> hidden, int type) {
        this.type = type;
        NeuralNetwork.INPUT_DIGITS_SIZE = n;
        switch(type) {
            case DIGITS : {
                NeuralNetwork.OUTPUT_DIGITS_SIZE = 10;
                break;
            }
            default : {
                NeuralNetwork.OUTPUT_DIGITS_SIZE = 2;
                break;
            }
        }
        images = new ArrayList<>();
        this.n = n;
        this.a = a;
        this.b = b;
        this.threshold = threshold;
        load_Neural_Net(hidden);
    }

    private void load_Neural_Net(List<Integer> hidden) {
        String dir = "";
        switch(type) {
            case FACES : {
                dir = NN_FACES_DIR;
                break;
            }
            default : {
                dir = NN_DIGITS_DIR;
                break;
            }
        }
        NeuralNetwork nn = loadNN(dir + "n:" + n + "_a:" + a + "_b:" + b + "_d:" + threshold);
        if(nn != null) {
            this.nn = nn;
        }
        else {
            this.nn = new NeuralNetwork(NeuralNetwork.INPUT_DIGITS_SIZE, hidden, NeuralNetwork.OUTPUT_DIGITS_SIZE);
        }
    }

     // Method to save a Person object to a file
     private void saveNN(NeuralNetwork nn) {
        String dir = "";
        switch(type) {
            case FACES : {
                dir = NN_FACES_DIR;
                break;
            }
            default : {
                dir = NN_DIGITS_DIR;
            }
        }
        String filename = dir + "n:" + n + "_a:" + a + "_b:" + b + "_d:" + threshold;
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
        switch (type) {
            case DIGITS : {
                loadImages(Util.DIGIT_VALIDATION_DATA);
                loadLabels(Util.DIGIT_VALIDATION_LABELS);
                break;
            }
            default : {
                loadImages(Util.FACE_VALIDATION_DATA);
                loadLabels(Util.FACE_VALIDATION_LABELS);
                break;
            }
        }
        Image image = images.get(0);
        System.out.println(image.toString());
        double[] output = nn.forward_propagation(image);
        System.out.println("Model gives");
        for(int i = 0; i < output.length; i++) {
            System.out.println(i + " : " + output[i]);
        }
        nn.back_propagate(image, output);
    }

    public void validate() {
        switch (type) {
            case DIGITS : {
                loadImages(Util.DIGIT_VALIDATION_DATA);
                loadLabels(Util.DIGIT_VALIDATION_LABELS);
                break;
            }
            default : {
                loadImages(Util.FACE_VALIDATION_DATA);
                loadLabels(Util.FACE_VALIDATION_LABELS);
                break;
            }
        }
        int correct = 0;
        for(int i = 0; i < images.size(); i++) {
            Image image = images.get(i);
            int answer = nn.fire(image);
            // System.out.println(image);
            // System.out.println(answer);
            if(answer == labels[image.getID()]) {
                correct++;
            }

        }
        // System.out.println(images.size());
        double accuracy = (double)(correct)/(double)(images.size());
        this.acc = accuracy;
        System.out.println("Correct " + correct + " out of " + images.size() + " for an accuracy of " + accuracy);
    }

    public void train() {
        // switch (type) {
        //     case DIGITS : {
        //         loadImages(Util.DIGIT_TRAINING_DATA);
        //         loadLabels(Util.DIGIT_TRAINING_LABELS);
        //         break;
        //     }
        //     default : {
        //         loadImages(Util.FACE_TRAINING_DATA);
        //         loadLabels(Util.FACE_TRAINING_LABELS);
        //         break;
        //     }
        // }
        trainingSet(threshold);
        long start = System.currentTimeMillis();
        long cnt = 0;
        while(cnt < TRAINING_CNT) {
            if(cnt % (TRAINING_CNT/10) == 0) {
                System.out.println(cnt);
            }

            for (int i = 0; i < trainingset.size(); i++) {
                Image image = trainingset.get(i);
                nn.train(image);
            }
            cnt++;
        }
        time = System.currentTimeMillis() - start;
        saveNN(this.nn);
    }
    public void test() {
        switch (type) {
            case DIGITS : {
                loadImages(Util.DIGIT_TEST_DATA);
                loadLabels(Util.DIGIT_TEST_LABELS);
                break;
            }
            default : {
                loadImages(Util.FACE_TEST_DATA);
                loadLabels(Util.FACE_TEST_LABELS);
                break;
            }
        }
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

    public void randomizeWeights() {
        nn.randomizeWeights();
    }

    public void outputTraining() {
        FileWriter writer = null;
        try {
            switch(type) {
                case FACES : {
                    writer = new FileWriter(FACE_TRAINING_OUTPUT_DIR + "n:" + n + "_a:" + a + "_b:" + b + "_d:" + threshold);
                    break;
                }
                case DIGITS : {
                    writer = new FileWriter(DIGIT_TRAINING_OUTPUT_DIR + "n:" + n + "_a:" + a + "_b:" + b + "_d:" + threshold);
                    break;
                }
            
            }
            writer.write("-------------------------\n");
            writer.write("Training Output File\n");
            writer.write("-------------------------\n");
            writer.write("\n");
            switch(type) {
                case FACES : {
                    writer.write("Faces\n\n");
                    break;
                }
                case DIGITS : {
                    writer.write("Digits\n\n");
                    break;
                }
            }
            writer.write("N:" + n + "\nA:" + a + "\nB:" + b + "\nTraining Threshold : " + threshold);
            writer.write("\n\n");
            writer.write("Trained " + TRAINING_CNT + " epochs\n");
            writer.write("Training Time : " + time + " ms\n");
            writer.write("Training Time : " + Util.millisecondsToHMS(time) + " HH:MM:SS\n");
            String formattedAcc = String.format("%.2f", acc * 100);
            writer.write("\nValidation Accuracy : " + formattedAcc + "%");
            writer.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
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
                Image image = new Image(n, a, b, file, type);
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
        return (int)(100000*(this.acc - other.acc));
    }
}
