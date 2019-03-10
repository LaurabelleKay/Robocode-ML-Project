package fnl;

public class RL {

    static int numActions = 4;
    static int energies = 2;
    static int distances = 3;
    static int numStates = energies * distances;
    static int c = 0;

    static int[][] states = new int[distances][energies];
    // static int[] states = new int[numStates];
    static int[] actions = new int[numActions];
    static double[][] Q = new double[numStates][numActions];

    static double alpha = 0.1;
    static double gamma = 0.9;
    static double epsilon = 0.8;
    static double reward = 0;

    public RL() {
    }

    public static void initSAQ() {
        int x = 0;

        for (int i = 0; i < distances; i++) {
            for (int j = 0; j < energies; j++) {
                states[i][j] = x++;
            }
        }

        for (int i = 0; i < numStates; i++) {
            for (int j = 0; j < numActions; j++) {
                Q[i][j] = 0.0;
            }
        }
    }

    public static int selectAction(int state) {
        int action = 0;
        double prob = Math.random();
        if (prob > epsilon) {
            action = maxAction(state);
        } else {
            action = (int) (Math.random() * (numActions));
        }
        return action;
    }

    public static void getQ(int currentState, int nextState, int action) {
        double q = maxQ(nextState);
        Q[currentState][action] += alpha * (reward + gamma * q - Q[currentState][action]);
    }

    public static void getSarsa(int s1, int a1, int s2, int a2, double r) {
        Q[s1][a1] += alpha * (r + gamma * Q[s2][a2] - Q[s1][a1]);
    }

    public static double maxQ(int state) {
        double val = 0;
        for (int i = 0; i < numActions; i++) {
            if (Q[state][i] > val) {
                val = Q[state][i];
            }
        }
        return val;
    }

    public static int maxAction(int state) {
        double val = 0.0;
        int act = 0;
        for (int i = 0; i < numActions; i++) {
            if (Q[state][i] > val) {
                val = Q[state][i];
                act = i;
            }
        }
        return act;
    }
}