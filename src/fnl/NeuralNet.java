package fnl;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class NeuralNet implements NeuralNetInterface {

    double[] targets;
    double[][] inputs;

    // double[] weights;
    double[][] hidWeights;
    double[][] hidWeightsOld;
    double[] outWeights;
    double[] outWeightsOld;

    // Input and output for each neuron
    double[] neurX;
    double[] neurY;

    // Derivatives
    double[] dEdY;
    double[] dEdX;
    double[][] dEdW;
    double[][] deltaH;
    double[] deltaO;
    double[] f_X;

    double error;
    int epochs;
    int pNum; // num in a pattern
    int layers;
    int numNeurons;

    int numIn;
    int numHid;
    double learnRate;
    double momentumTerm;
    double a;
    double b;
    int numweights;

    public NeuralNet(int argNumInputs, int argNumHidden, double argLearningRate, double argMomentumTerm, double argA,
            double argB) {
        numIn = argNumInputs;
        numHid = argNumHidden;
        learnRate = argLearningRate;
        momentumTerm = argMomentumTerm;
        a = argA;
        b = argB;
    }

    public void initializeNet() {

        pNum = 24; // number of patterns
        numNeurons = numIn + numHid + 1;
        layers = 3;

        numweights = numHid * (numIn + 1 + 1) + 1;
        hidWeights = new double[numHid][numIn + 1];
        hidWeightsOld = new double[numHid][numIn + 1];

        outWeights = new double[numHid + 1];
        outWeightsOld = new double[numHid + 1];

        deltaH = new double[numHid][numIn + 1];
        deltaO = new double[numHid + 1];

        inputs = new double[pNum][numIn];
        targets = new double[pNum];

        // inputs = new double[][] { { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0,
        // 1.0 } };
        // targets = new double[] { 0.0, 1.0, 1.0, 0.0 };

        // inputs = new double[][] { { -1.0, -1.0 }, { -1.0, 1.0 }, { 1.0, -1.0 }, {
        // 1.0, 1.0 } };
        // targets = new double[] { -1.0, 1.0, 1.0, -1.0 };

        neurX = new double[numNeurons];
        neurY = new double[numNeurons];

        dEdX = new double[numNeurons];
        dEdY = new double[numNeurons];
        f_X = new double[numNeurons];

        initializeWeights();
    }

    public void run() throws IOException {
        double totErr = 0.0;
        double[] X = new double[numIn];
        double err;
        epochs = 0;

        // String file = new String("D:/Laurabelle/Documents/VSCODE/robProj/Weights");
        File weights = new File("robProj/Weights");

        // load(file);

        while (true) {
            epochs++;
            totErr = 0.0;
            for (int i = 0; i < pNum; i++) {
                X = inputs[i];
                err = train(X, targets[i]);
                totErr += err;
                backward(targets[i]);
            }
            // double rmse = Math.sqrt(totErr/pNum);

            // if(epochs % 1 == 0) { System.out.printf("%f\n",totErr); }
            if (epochs == 5000) {
                System.out.printf("%d\n", epochs);
                System.out.printf("Final error: %f\n", totErr);
                save(weights);
                break;
            }
        }
    }

    public double sigmoid(double x) {
        double val = (2 / (1 + Math.exp(-x))) - 1;
        return val;
    }

    public double customSigmoid(double x) {
        double val = ((b - a) / (1 + Math.exp(-x))) - (-a);
        return val;
    }

    public double customSigmoid(double x, double a, double b) {
        double val = ((b - a) / (1 + Math.exp(-x))) - (-a);
        return val;
    }

    public void initializeWeights() {
        for (int i = 0; i < numHid; i++) {
            outWeights[i] = Math.random() - 0.5;
            outWeightsOld[i] = outWeights[i];
            for (int j = 0; j < numIn + 1; j++) {
                hidWeights[i][j] = Math.random() - 0.5;
                hidWeightsOld[i][j] = hidWeights[i][j];
            }
        }
        outWeights[numHid] = Math.random() - 0.5;
        outWeightsOld[numHid] = outWeights[numHid];
    }

    public void zeroWeights() {
        for (int i = 0; i < numHid; i++) {
            outWeights[i] = 0.0;
            outWeightsOld[i] = 0.0;
            for (int j = 0; j < numIn; j++) {
                hidWeights[i][j] = 0.0;
                hidWeightsOld[i][j] = 0.0;
            }
        }
        outWeights[numHid] = 0.0;
        outWeightsOld[numHid] = 0.0;
    }

    public void forward(double[] X) {

        // for input neurons
        double[] y1 = new double[numIn + 1];
        for (int j = 0; j < numIn; j++) {
            neurX[j] = X[j];
            neurY[j] = neurX[j];
            y1[j] = neurY[j];
        }
        y1[numIn] = bias;

        int J;
        double[] w;

        // for hidden neurons
        double[] y2 = new double[numHid + 1];
        for (int j = numIn; j < numIn + numHid; j++) {
            J = numIn + 1;
            w = new double[J];
            w = hidWeights[j - numIn];

            // Inputs to neuron
            neurX[j] = weightedSum(J, w, y1);

            // Output of neuron
            neurY[j] = sigmoid(neurX[j]);

            y2[j - numIn] = neurY[j];
        }
        y2[numHid] = bias;

        // for output neuron
        J = numHid + 1;
        int n = numIn + numHid;
        neurX[n] = weightedSum(J, outWeights, y2);
        // neurY[n] = neurX[n];
        // neurY[n] = customSigmoid(neurX[n]);
    }

    public void backward(double target) {

        // for output neuron
        int n = numIn + numHid;

        double Y;
        double[] w;

        // Final output
        Y = neurY[n];

        // Y'
        // double d = (1/(51.0 - 39.0)) * (1.0 - (Y * Y));
        // double d = b * Y * (1.0 - Y);
        double d = Y;
        double delta = (target - Y) * d;

        double[] X = new double[numHid + 1];

        for (int i = 0; i < numHid; i++) {
            X[i] = neurY[i + numIn];
        }
        X[numHid] = bias;

        updateWeights(outWeights, delta, X, outWeightsOld);

        // for hidden neurons
        w = outWeights;
        int jj;
        double[] deltah = new double[numHid];
        for (int j = numIn; j < numIn + numHid; j++) {
            jj = j - numIn;

            Y = neurY[j];
            // d = 0.5 * (1.0 - (Y * Y));
            d = Y * (1.0 - Y);
            deltah[jj] = w[jj] * delta * d;
        }

        X = new double[numIn + 1];
        for (int i = 0; i < numIn; i++) {
            X[i] = neurY[i];
        }
        X[numIn] = bias;

        for (int i = 0; i < numHid; i++) {
            updateWeights(hidWeights[i], deltah[i], X, hidWeightsOld[i]);
        }
    }

    // Update the old and new weights
    public void updateWeights(double[] W1, double delta, double[] X, double[] W2) {
        double d;
        double[] temp = new double[W1.length];
        for (int i = 0; i < W1.length; i++) {
            temp[i] = W1[i];
            d = W1[i] - W2[i];
            W1[i] = W1[i] + (learnRate * delta * X[i]) + (d * momentumTerm);
            W2[i] = temp[i];
        }
    }

    public double weightedSum(int J, double[] w, double[] y) {
        double x = 0;
        for (int j = 0; j < J; j++) {
            x += w[j] * y[j];
        }
        return x;
    }

    public double outputFor(double[] X) {
        forward(X);
        double out = neurY[numIn + numHid];
        return out;
    }

    public double train(double[] X, double argValue) {
        double out = outputFor(X);
        double err = 0.5 * ((argValue - out) * (argValue - out));
        return err;
    }

    public double[] getWeights() {
        double[] W = new double[numweights];
        int x = 0;
        for (int i = 0; i < outWeights.length; i++) {
            W[x] = outWeights[i];
            x++;
        }
        for (int i = 0; i < numHid; i++) {
            for (int j = 0; j < numIn; j++) {
                W[x] = hidWeights[i][j];
                x++;
            }
        }
        return W;
    }

    public void setWeights(double[] W) {
        int x = 0;
        for (int i = 0; i < outWeights.length; i++) {
            outWeights[i] = W[x];
            x++;
        }
        for (int i = 0; i < numHid; i++) {
            for (int j = 0; j < numIn; j++) {
                hidWeights[i][j] = W[x];
                x++;
            }
        }
    }

    public void QtoNN() {
        int c = 0;
        for (int i = 0; i < RL.numStates; i++) {
            for (int j = 0; j < RL.numActions; j++) {
                targets[c] = RL.Q[i][j];
                c++;
            }
        }

        int cc = 0;
        for (int i = 0; i < RL.distances; i++) {
            for (int j = 0; j < RL.energies; j++) {
                for (int k = 0; k < RL.numActions; k++) {
                    inputs[cc][0] = i;
                    inputs[cc][1] = j;
                    inputs[cc][2] = k;
                    cc++;
                }
            }
        }
    }

    public void saveNN() throws IOException {
        File ins = new File("robProj/NNInput");
        FileOutputStream out = new FileOutputStream(ins);
        DataOutputStream dos = new DataOutputStream(out);
        for (int i = 0; i < pNum; i++) {
            dos.writeDouble(inputs[i][0]);
            dos.writeDouble(inputs[i][1]);
            dos.writeDouble(inputs[i][2]);
        }
        dos.close();
        File outs = new File("robProj/NNTarget");
        out = new FileOutputStream(outs);
        DataOutputStream ds = new DataOutputStream(out);
        for (int i = 0; i < pNum; i++) {
            ds.writeDouble(targets[i]);
        }
        ds.close();
    }

    public void loadNN() throws IOException {
        String fileName = new String("robProj/NNTarget");
        FileInputStream in = new FileInputStream(fileName);
        DataInputStream dis = new DataInputStream(in);
        int i = 0;
        while (dis.available() > 0) {
            if (i == pNum) {
                break;
            }
            targets[i] = dis.readDouble();
            // targets[i] = (targets[i] - 39.0) * 3/(51.0 - 39.0);
            i++;
        }
        dis.close();
        fileName = new String("robProj/NNInput");
        in = new FileInputStream(fileName);
        DataInputStream ds = new DataInputStream(in);
        i = 0;
        while (ds.available() > 0) {
            if (i == pNum) {
                break;
            }
            inputs[i][0] = ds.readDouble();
            inputs[i][1] = ds.readDouble();
            inputs[i][2] = ds.readDouble();
            i++;
        }
        ds.close();
    }

    public void save(File argFile) throws IOException {
        double[] W = getWeights();
        FileOutputStream out = new FileOutputStream(argFile);
        DataOutputStream dos = new DataOutputStream(out);
        for (int i = 0; i < W.length; i++) {
            dos.writeDouble(W[i]);
        }
        dos.close();
    }

    public void load(String argFileName) throws IOException {
        double[] W = new double[numweights];
        FileInputStream in = new FileInputStream(argFileName);
        DataInputStream dis = new DataInputStream(in);
        int i = 0;
        while (dis.available() > 0) {
            W[i] = dis.readDouble();
            i++;
        }
        dis.close();
        setWeights(W);
    }
}