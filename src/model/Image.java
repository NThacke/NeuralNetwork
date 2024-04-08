package model;

import java.io.*;
import model.util.*;

public class Image extends AbstractImage {

    public Image(int n, int a, int b, RandomAccessFile file) {
        this.n = n;
        this.a = a;
        this.b = b;
        image = new char[Util.DIGIT_IMAGE_LENGTH][Util.DIGIT_IMAGE_WIDTH];
        try {
            for(int i = 0; i < image.length; i++) {
                for(int j = 0; j < image.length; j++) {
                    int c = file.read();
                    if(c != -1) {
                        image[i][j] = (char)(c);
                    }
                }
            }
            file.readLine();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        calc_phi(n, a, b);
    }

    protected boolean validDimensions(int n, int a, int b) {
        if(n * a * b != 28 * 28) {
            throw new IllegalArgumentException("n*a*b must equal 784");
        }
        if(28%a != 0) {
            throw new IllegalArgumentException("a must divide 28");
        }
        if(28%b != 0) {
            throw  new IllegalArgumentException("b must divide 28");
        }
        return true;
    }
}
