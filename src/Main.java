import model.*;

public class Main {
    public static void main(String[] args) {


        Driver d = new Driver(196, 2, 2, 1.0);
//        d.randomizeWeights();
//        d.train();
        d.load();
        d.validate();
//        d.nn.randomizeWeights();
//        d.back_prop_test();
//        d.validate();
//        d.testing();
    }
}