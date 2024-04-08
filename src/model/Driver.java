package model;

import model.util.Util;

import java.util.*;
import java.io.*;
public class Driver {

    NeuralNetwork nn;

    List<Image> images;

    private int[] labels;

    private int n;
    private int a;
    private int b;

    public Driver(int n, int a, int b, double threshold) {
        this.n = n;
        this.a = a;
        this.b = b;
        nn = new NeuralNetwork(n, 329, 10);

    }

    public void train() {
        images = new ArrayList<>();
        loadImages(Util.DIGIT_TRAINING_DATA);
        loadLabels(Util.DIGIT_TRAINING_LABELS);

        System.out.println(images.get(0));
        double[] arr = nn.forward_propogation(images.get(0));

        System.out.println(Arrays.toString(arr));
    }

    private void loadImages(String filename) {
        try {
            RandomAccessFile file = new RandomAccessFile(filename, "r");
            int id = 0;
            while(file.getFilePointer() < file.length()) {
                Image image = new Image(n, a, b, file);
                image.setID(id);
                images.add(image);
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
            labels = new int[images.size()];
            Scanner scanner = new Scanner(new File(filename));
            int i = 0;
            while(scanner.hasNext()) {
                labels[i] = scanner.nextInt();
                i++;
            }
            scanner.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
