import java.util.Comparator;
import java.util.*;

import model.*;

public class Main {

    private static final int[][] digit_dim = {{49, 4, 4,}, {196, 2, 2,}, {784, 1, 1}};
    public static void main(String[] args) {

        double best = 0.0;
        // Driver bestDriver = new Driver(784, 1, 1, 1.0);
        // bestDriver.load();
        // bestDriver.validate();
        List<Driver> list = new ArrayList<>();
        for(int i = 0; i<digit_dim.length; i++) {
            if(i == 2) {
                int n = digit_dim[i][0];
                int a = digit_dim[i][1];
                int b = a;
                for(int j = 7; j<=10; j++) {
                    Driver d = new Driver(n, a, b, (j/10.0));
                    // d.randomizeWeights();
                    d.load();
                    d.train();
                    d.validate();
                    if(d.acc > best) {
                        best = d.acc;
                        // bestDriver = d;
                    }
                    list.add(d);
                }
            }
        }
        Collections.sort(list);
        for(Driver d : list) {
            System.out.println(d);
        }
        // System.out.println("Best accuracy is " + best + " from driver " + bestDriver.toString());
    }
}