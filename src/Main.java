import java.io.File;
import java.io.RandomAccessFile;
import java.util.*;

import model.*;
import model.util.Util;

public class Main {

    private static final int[][] digit_dim = {{49, 4, 4,}, {196, 2, 2,}, {784, 1, 1}};

    private static final int[][] face_dim = {{42, 10, 10}, {168, 5, 5}, {1050, 2, 2}};
    public static void main(String[] args) {

        digitTraining();
        // faceTraining();
    }

    private static void digitTraining() {
        double best = 0.0;
        List<Driver> list = new ArrayList<>();
        List<Integer> hidden = hidden(5000);
        Driver bestDriver = null;
        
        for(int i = 0; i<digit_dim.length; i++) {
            int n = digit_dim[i][0];
            int a = digit_dim[i][1];
            int b = a;
            List<Image> images = loadImages(n, a, b, Util.DIGIT_TRAINING_DATA, Image.DIGITS);
            int[] labels = loadLabels(images, Util.DIGIT_TRAINING_LABELS);
            for(int j = 1; j<=10; j++) {
                    Driver d = new Driver(n, a, b, (j/10.0), hidden, Util.DIGITS);
                    d.labels = labels;
                    d.images = Util.copy(images);
                    d.randomizeWeights();
                    d.train();
                    d.validate();
                    d.outputTraining();
                    // d.test();
                    if(d.acc > best) {
                        best = d.acc;
                        bestDriver = d;
                    }
                    list.add(d);
            }
        }
        Collections.sort(list);
        for(Driver d : list) {
            System.out.println(d);
        }
        System.out.println("Best accuracy is " + best + " from driver " + bestDriver.toString());
    }

    private static void faceTraining() {
        double best = 0.0;
        List<Driver> list = new ArrayList<>();
        List<Integer> hidden = hidden(451);
        Driver bestDriver = null;
        
        for(int i = 0; i<face_dim.length; i++) {
            int n = face_dim[i][0];
            int a = face_dim[i][1];
            int b = a;
            List<Image> images = loadImages(n, a, b, Util.FACE_TRAINING_DATA, Image.FACES);
            int[] labels = loadLabels(images, Util.FACE_TRAINING_LABELS);
            for(int j = 1; j<=10; j++) {
                    Driver d = new Driver(n, a, b, (j/10.0), hidden, Util.FACES);
                    d.labels = labels;
                    d.images = Util.copy(images);
                    // d.randomizeWeights();
                    d.train();
                    d.validate();
                    // d.outputTraining();
                    // d.test();
                    if(d.acc > best) {
                        best = d.acc;
                        bestDriver = d;
                    }
                    list.add(d);
            }
        }
        Collections.sort(list);
        for(Driver d : list) {
            System.out.println(d);
        }
        System.out.println("Best accuracy is " + best + " from driver " + bestDriver.toString());
    }

    private static List<Integer> hidden(int sample_size) {
        // double h = sample_size / (2 * (NeuralNetwork.INPUT_DIGITS_SIZE + NeuralNetwork.OUTPUT_DIGITS_SIZE)); //recommend hidden neuron size
        double h = NeuralNetwork.INPUT_DIGITS_SIZE / 2;

        List<Integer> list = new ArrayList<>();
        list.add((int)(h));
        return list;
    }

    private static List<Image> loadImages(int n, int a, int b, String filename, int type) {
        try {
            List<Image> images = new ArrayList<>();
            RandomAccessFile file = new RandomAccessFile(filename, "r");
            int id = 0;
            while(file.getFilePointer() < file.length()) {
                Image image = new Image(n, a, b, file, type);
                image.setID(id);
                images.add(image);
                id++;
            }
            file.close();
            return images;
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private static int[] loadLabels(List<Image> images, String filename) {
        try {
            int[] labels = new int[images.size()];
            Scanner scanner = new Scanner(new File(filename));
            int i = 0;
            while(scanner.hasNext()) {
                labels[i] = scanner.nextInt();
                images.get(i).setLabel(labels[i]);
                i++;
            }
            scanner.close();
            return labels;
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}