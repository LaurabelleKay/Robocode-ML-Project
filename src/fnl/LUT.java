package fnl;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class LUT implements LUTInterface {
    int numInVect;
    int numIn;
    int[] floor;
    int[] ceil;

    static double[][] table = new double[RL.numActions * RL.numStates][2];

    public LUT(int argNumInputVec, int[] argVariableFloor, int[] argVariableCeiling, int argNumInputs) {
        numInVect = argNumInputVec;
        numIn = argNumInputs;
        floor = argVariableFloor;
        ceil = argVariableCeiling;
    }

    public void QtoLUT() {
        int s = RL.numStates;
        int a = RL.numActions;
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < a; j++) {
                table[a * i + j][1] = RL.Q[i][j];
            }
        }
    }

    @Override
    public void initialiseLUT() {
        int s = RL.numStates;
        int a = RL.numActions;

        for (int i = 0; i < s; i++) {
            for (int j = 0; j < a; j++) {
                table[a * i + j][0] = a * i + j;
                table[a * i + j][1] = 0.0;
            }
        }
    }

    @Override
    public int indexFor(double[] X) {
        int index = 0;
        double val = X[0];
        for (int i = 0; i < RL.numStates; i++) {
            for (int j = 0; j < RL.numActions; j++) {
                if (val == RL.Q[i][j]) {
                    index = RL.numActions * i + j;
                }
            }
        }
        return index;
    }

    @Override
    public double outputFor(double[] X) {
        int index = indexFor(X);
        double val = table[index][2];
        return val;
    }

    @Override
    public double train(double[] X, double argValue) {
        double val = outputFor(X);

        return val;
    }

    @Override
    public void save(File argFile) throws IOException {
        int s = RL.numStates;
        int a = RL.numActions;
        FileOutputStream out = new FileOutputStream(argFile);
        DataOutputStream dos = new DataOutputStream(out);
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < a; j++) {
                for (int k = 0; k < 2; k++) {
                    dos.writeDouble(table[a * i + j][k]);
                }
            }
        }
        dos.close();
    }

    @Override
    public void load(String argFileName) throws IOException {
        int s = RL.numStates;
        int a = RL.numActions;
        double[] buf = new double[s * a * 2];
        FileInputStream in = new FileInputStream(argFileName);
        DataInputStream dis = new DataInputStream(in);
        int c = 0;
        while (dis.available() > 0) {
            buf[c] = dis.readDouble();
            c++;
        }
        dis.close();
        c = 0;
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < a; j++) {
                for (int k = 0; k < 2; k++) {
                    table[a * i + j][k] = buf[c];
                    c++;
                }
            }
        }
    }
}