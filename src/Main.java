import java.io.File;
import java.io.RandomAccessFile;
import java.util.*;

import model.*;
import model.util.Util;

public class Main {

    private static final int[][] digit_dim = {{49, 4, 4,}, {196, 2, 2,}, {784, 1, 1}};
    public static void main(String[] args) {

        double best = 0.0;
        List<Driver> list = new ArrayList<>();
        List<Integer> hidden = hidden();
        Driver bestDriver = new Driver(196, 2, 2, 1.0, hidden);
        bestDriver.load();
        // bestDriver.randomizeWeights();
        bestDriver.train();
        bestDriver.validate();
        
        // for(int i = 1; i<digit_dim.length; i++) {
        //     int n = digit_dim[i][0];
        //     int a = digit_dim[i][1];
        //     int b = a;
        //     List<Image> images = loadImages(n, a, b, Util.DIGIT_TRAINING_DATA);
        //     int[] labels = loadLabels(images, Util.DIGIT_TRAINING_LABELS);
        //     for(int j = 10; j<=10; j++) {
        //             Driver d = new Driver(n, a, b, (j/10.0), hidden);
        //             d.images = images;
        //             d.labels = labels;
        //             // d.randomizeWeights();
        //             d.load();
        //             d.train();
        //             d.validate();
        //             if(d.acc > best) {
        //                 best = d.acc;
        //                 bestDriver = d;
        //             }
        //             list.add(d);
        //     }
        // }
        // Collections.sort(list);
        // for(Driver d : list) {
        //     System.out.println(d);
        // }
        // System.out.println("Best accuracy is " + best + " from driver " + bestDriver.toString());
    }

    private static List<Integer> hidden() {
        List<Integer> list = new ArrayList<>();
        list.add(32);
        return list;
    }

    private static List<Image> loadImages(int n, int a, int b, String filename) {
        try {
            List<Image> images = new ArrayList<>();
            RandomAccessFile file = new RandomAccessFile(filename, "r");
            int id = 0;
            while(file.getFilePointer() < file.length()) {
                Image image = new Image(n, a, b, file);
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