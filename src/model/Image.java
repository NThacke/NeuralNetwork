package model;

import java.io.*;
import java.util.Arrays;
import java.util.*;

import model.util.*;

public class Image extends AbstractImage {

    public Image(int n, int a, int b, RandomAccessFile file, int type) {
        this.type = type;
        this.n = n;
        this.a = a;
        this.b = b;
        if(type == FACES) {
            image = new char[Util.FACE_IMAGE_LENGTH][Util.FACE_IMAGE_WIDTH];
        }
        else {
            image = new char[Util.DIGIT_IMAGE_LENGTH][Util.DIGIT_IMAGE_WIDTH];
        }
        try {
            if(type == FACES) {
                file.readLine();
            }
            for(int i = 0; i < image.length; i++) {
                for(int j = 0; j < image[i].length; j++) {
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
        if(type == DIGITS) {
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
        else {
            if(n * a * b != 70 * 60) {
                throw new IllegalArgumentException("n*a*b must equal 4200");
            }
            if(70%a != 0) {
                throw new IllegalArgumentException("a must divide 70");
            }
            if(60%b != 0) {
                throw  new IllegalArgumentException("b must divide 60");
            }
            return true;
        }
    }
}
