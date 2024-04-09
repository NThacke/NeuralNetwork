import model.*;

public class Main {

    private static final int[][] digit_dim = {{49, 4, 4,}, {169, 2, 2,}, {784, 1, 1}};
    public static void main(String[] args) {

        double best = 0.0;
        Driver bestDriver = null;

        for(int i = 0; i<digit_dim.length; i++) {
            int n = digit_dim[i][0];
            int a = digit_dim[i][1];
            int b = a;
            for(int j = 1; j<=10; j++) {
                Driver d = new Driver(n, a, b, (j/10.0));
                d.randomizeWeights();
                d.train();
                d.validate();
                if(d.acc > best) {
                    best = d.acc;
                    bestDriver = d;
                }
            }
        }
        System.out.println("Best accuracy is " + best + " from driver " + bestDriver.toString());
    }
}