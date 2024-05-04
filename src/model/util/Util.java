package model.util;

import java.util.*;
import java.text.SimpleDateFormat;
import model.*;


public abstract interface Util {

    public static final int FACES = 19283;
    public static final int DIGITS = 12920202;

    public static final int FACE_IMAGE_LENGTH = 70;
    public static final int FACE_IMAGE_WIDTH = 60;

    public static final int DIGIT_IMAGE_LENGTH = 28;
    public static final int DIGIT_IMAGE_WIDTH = 28;

    public static final String DIGITS_HIDDEN_LAYER_WEIGHTS_DIR = "src/data/weights/digits/hidden/";
    public static final String DIGITS_OUTPUT_LAYER_WEIGHTS_DIR = "src/data/weights/digits/output/";

    public static final String DIGIT_0_DIRECTORY = "src/data/weights/digits/0/";
    public static final String DIGIT_1_DIRECTORY = "src/data/weights/digits/1/";
    public static final String DIGIT_2_DIRECTORY = "src/data/weights/digits/2/";
    public static final String DIGIT_3_DIRECTORY = "src/data/weights/digits/3/";
    public static final String DIGIT_4_DIRECTORY = "src/data/weights/digits/4/";
    public static final String DIGIT_5_DIRECTORY = "src/data/weights/digits/5/";
    public static final String DIGIT_6_DIRECTORY = "src/data/weights/digits/6/";
    public static final String DIGIT_7_DIRECTORY = "src/data/weights/digits/7/";
    public static final String DIGIT_8_DIRECTORY = "src/data/weights/digits/8/";
    public static final String DIGIT_9_DIRECTORY = "src/data/weights/digits/9/";

    public static final String DIGIT_OUTPUT_TRAINING_DIRECTORY = "src/data/output/digits/training/";
    public static final String DIGIT_OUTPUT_DEMO_DIRECTORY = "src/data/output/digits/demo/";

    public static final String FACES_OUTPUT_TRAINING_DIRECTORY = "src/data/output/faces/training/";
    public static final String FACES_OUTPUT_DEMO_DIRECTORY = "src/data/output/faces/demo/";

    public static final String DIGIT_TRAINING_DATA = "src/data/digitdata/trainingimages";
    public static final String DIGIT_TEST_DATA = "src/data/digitdata/testimages";
    public static final String DIGIT_VALIDATION_DATA = "src/data/digitdata/validationimages";

    public static final String DIGIT_TRAINING_LABELS = "src/data/digitdata/traininglabels";
    public static final String DIGIT_TEST_LABELS = "src/data/digitdata/testlabels";
    public static final String DIGIT_VALIDATION_LABELS = "src/data/digitdata/validationlabels";

    public static final String FACE_TRAINING_DATA = "src/data/facedata/facedatatrain";
    public static final String FACE_VALIDATION_DATA = "src/data/facedata/facedatavalidation";
    public static final String FACE_TEST_DATA = "src/data/facedata/facedatatest";

    public static final String FACE_TRAINING_LABELS = "src/data/facedata/facedatatrainlabels";
    public static final String FACE_VALIDATION_LABELS = "src/data/facedata/facedatavalidationlabels";
    public static final String FACE_TEST_LABELS = "src/data/facedata/facedatatestlabels";

    public static final String FACE_WEIGHTS_DIRECTORY = "src/data/weights/faces/";

    public static final String FACE_TRAINING_OUTPUT_DIR = "src/data/output/training/faces/";
    public static final String DIGIT_TRAINING_OUTPUT_DIR = "src/data/output/training/digits/";



    public static final long seed = 1;
    public static final Random random = new Random();


    public static double sigmoid(double x) {
        double e = Math.exp(-x);
        return 1/(1+e);
    }

    public static String millisecondsToHMS(long milliseconds) {
        long seconds = milliseconds / 1000;
        long hours = seconds / 3600;
        long minutes = (seconds % 3600) / 60;
        seconds = seconds % 60;

        return String.format("%02d:%02d:%02d", hours, minutes, seconds);
    }

    public static String hhmmss(long milliseconds) {
        SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss");
        sdf.setTimeZone(TimeZone.getTimeZone("GMT")); // Setting timezone to GMT to ensure accurate calculation
        String formattedTime = sdf.format(new Date(milliseconds));
        return formattedTime;
    }

    public static double[] toDoubleArray(int[] arr) {
        double[] out = new double[arr.length];
        for(int i = 0; i< arr.length; i++) {
            out[i] = (double)(arr[i]);
        }
        return out;
    }

    public static List<Image> copy(List<Image> list) {
        List<Image> copy = new ArrayList<>();
        for(Image i : list) {
            copy.add(i);
        }
        return copy;
    }

    /**
     * Calculates the standard deviation of a given array of values.
     * 
     * @param values
     * @return
     */
    public static double stddv(double[] values) {
        double n = values.length;
        
        double mean = mean(values);
        
        // Step 2: Calculate the sum of squared differences from the mean
        double sumSquaredDifferences = 0;
        for (double value : values) {
            sumSquaredDifferences += Math.pow(value - mean, 2);
        }
        
        // Step 3: Calculate the variance
        double variance = sumSquaredDifferences / n;
        
        // Step 4: Calculate the standard deviation (square root of the variance)
        double standardDeviation = Math.sqrt(variance);
        
        return standardDeviation;
    }

    public static double mean(double[] values) {
        double n = values.length;
        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        double mean = sum / n;
        return mean;
    }
}
