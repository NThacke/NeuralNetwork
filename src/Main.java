import java.io.File;
import java.io.FileWriter;
import java.io.RandomAccessFile;
import java.util.*;

import model.*;
import model.util.Util;

public class Main {

    private static final int[][] digit_dim = {{49, 4, 4,}, {196, 2, 2,}, {784, 1, 1}};

    private static final int[][] face_dim = {{42, 10, 10}, {168, 5, 5}, {1050, 2, 2}};
    public static void main(String[] args) {
        // faceDemo();
        // digitDemo();
        digitStats();
        faceStats();
    }

    private static void faceDemo() {
        Driver d = new Driver(168, 5, 5, 1.0, hidden(5000), Util.FACES);
        d.validate();
    }
    private static void digitDemo() {
        Driver d = new Driver(196, 2, 2, 0.9, hidden(5000), Util.DIGITS);
        d.validate();
    }

    private static void digitStats() {
        try {
            FileWriter writer = new FileWriter("src/digits_prediction_errors.txt");
            for(int i = 0; i < digit_dim.length; i++ ) {
                int n = digit_dim[i][0];
                int a = digit_dim[i][1];
                int b = a;
                for(int j = 1; j <= 10; j++) {
                    Driver d = new Driver(n, a, b, (j/10.0), hidden(3000), Util.DIGITS);
                    double [][] stats = d.stats(j/10.0);
                    double[] prediction_errors = stats[1];
                    double[] arr = stats[0];
                    writer.write(d.toString() + "\n");
                    writer.write("Prediction errors : ");
                    for(int k = 0; k < prediction_errors.length; k++) {
                        writer.write(prediction_errors[k] + "\t");
                    }
                    writer.write("\n");
                    writer.write("Mean : " + arr[0] + "\n");
                    writer.write("Sigma : " + arr[1] + "\n\n");
                }
            }
            writer.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

    private static void faceStats() {
        try {
            FileWriter writer = new FileWriter("src/faces_prediction_errors.txt");
            for(int i = 0; i < face_dim.length; i++ ) {
                int n = face_dim[i][0];
                int a = face_dim[i][1];
                int b = a;
                for(int j = 1; j <= 10; j++) {
                    Driver d = new Driver(n, a, b, (j/10.0), hidden(3000), Util.FACES);
                    double [][] stats = d.stats(j/10.0);
                    double[] prediction_errors = stats[1];
                    double[] arr = stats[0];
                    writer.write(d.toString() + "\n");
                    writer.write("Prediction errors : ");
                    for(int k = 0; k < prediction_errors.length; k++) {
                        writer.write(prediction_errors[k] + "\t");
                    }
                    writer.write("\n");
                    writer.write("Mean : " + arr[0] + "\n");
                    writer.write("Sigma : " + arr[1] + "\n\n");
                }
            }
            writer.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

    private static void digitTest() {
        List<Integer> hidden = hidden(5000);
        try {
            FileWriter writer = new FileWriter("src/fileoutput.txt");
        for(int i = 0; i<digit_dim.length; i++) {
            int n = digit_dim[i][0];
            int a = digit_dim[i][1];
            int b = a;
                for(int j = 1; j<=10; j++) {
                        Driver d = new Driver(n, a, b, (j/10.0), hidden, Util.DIGITS);
                        d.test();
                        writer.append(d.toString() + "\n");
                        writer.append(String.valueOf(d.acc) + "\n\n");
                }
                
            }
            writer.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

    private static void digitTraining() {
        double best = 0.0;
        List<Driver> list = new ArrayList<>();
        Driver bestDriver = null;
        
        for(int i = 0; i<digit_dim.length; i++) {
            int n = digit_dim[i][0];
            int a = digit_dim[i][1];
            int b = a;
            NeuralNetwork.INPUT_DIGITS_SIZE = n;
            List<Integer> hidden = hidden(5000);

            List<Image> images = loadImages(n, a, b, Util.DIGIT_TRAINING_DATA, Image.DIGITS);
            int[] labels = loadLabels(images, Util.DIGIT_TRAINING_LABELS);
            for(int j = 1; j<=10; j++) {
                    
                    Driver d = new Driver(n, a, b, (j/10.0), hidden, Util.DIGITS);
                    d.labels = labels;
                    d.images = Util.copy(images);
                    d.randomizeWeights();
                    d.train();
                    d.validate();
                    d.test();
                    d.outputTraining();
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
        Driver bestDriver = null;
        
        for(int i = 0; i<face_dim.length; i++) {
            int n = face_dim[i][0];
            int a = face_dim[i][1];
            int b = a;

            NeuralNetwork.INPUT_DIGITS_SIZE = n;
            List<Integer> hidden = hidden(5000);

            List<Image> images = loadImages(n, a, b, Util.FACE_TRAINING_DATA, Image.FACES);
            int[] labels = loadLabels(images, Util.FACE_TRAINING_LABELS);
            for(int j = 1; j<=10; j++) {
                    Driver d = new Driver(n, a, b, (j/10.0), hidden, Util.FACES);
                    d.labels = labels;
                    d.images = Util.copy(images);
                    d.randomizeWeights();
                    d.train();
                    d.validate();
                    d.test();
                    d.outputTraining();

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