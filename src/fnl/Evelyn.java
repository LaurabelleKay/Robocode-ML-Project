package fnl;

import java.io.PrintStream;

import java.io.IOException;
import java.io.File;

import robocode.AdvancedRobot;
import robocode.BattleEndedEvent;
import robocode.BulletHitEvent;
import robocode.DeathEvent;
import robocode.HitByBulletEvent;
import robocode.HitRobotEvent;
import robocode.RobocodeFileOutputStream;
import robocode.RoundEndedEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;
import robocode.util.Utils;

public class Evelyn extends AdvancedRobot {
    static boolean start = true;
    static boolean sL = true;
    static boolean sLearn = true;

    static int numRounds = 0;
    static int roundNum = 0;
    static double[] winRate;
    static int nR = 0;
    static double[] wActions;
    static int count = 0;
    static double[] totR;

    double bearing;
    double distance;
    double enemyEnergy;
    double robotEnergy;
    double heading;
    double radarDir;

    static double numCorrectActions = 0.0;
    static double numWrongActions = 0.0;
    static double numActions = 0.0;
    static double wins = 0.0;

    static int currentState;
    static int nextState;
    static int action;
    static int nextAction;
    static int state;
    static double totalReward;

    static int s1 = 0;
    static int s2 = 0;
    static int s3 = 0;

    static int a1 = 0;
    static int a2 = 0;
    static int a3 = 0;

    static double r1 = 0;
    static double r2 = 0;

    static LUT l;
    static NeuralNet n;

    public void run() {
        if (start) {
            RL.initSAQ();
            int[] fl = new int[] { 0, 0 };
            int[] cl = new int[] { 100, 100 };
            l = new LUT(1, fl, cl, RL.numActions * RL.numStates);
            l.initialiseLUT();
            start = false;
            numRounds = getNumRounds();
            totR = new double[numRounds / 25];
            // System.setProperty("NOSECURITY", "true");

            //Function approximation with the neural net
            n = new NeuralNet(3, 10, 0.2, 0.0, -1.0, 1.0);
            n.initializeNet();

            try {
                n.loadNN();
                n.run();
            } catch (IOException e) {
                System.out.println("IOEXCEPTION");
            }

            // String argFileName = getDataDirectory();
            /* File file = getDataFile("LUTE0.2.txt"); */

            /*
             * String argFileName = new
             * String("robProj/LUT"); try {
             * l.load(argFileName); } catch (IOException e) {
             * System.out.println("IOEXCEPTION"); }
             */

        }
        while (true) {
            turnRadarRight(Double.POSITIVE_INFINITY);
        }
    }

    public void hardCode() {
        state = getState(distance);
        switch (state) {
        case 0:
            action = 2;
            break;

        case 1:
            action = 1;
            break;

        case 2:
            action = 1;
            break;

        case 3:
            action = 3;
            break;

        case 4:
            action = 0;
            break;

        case 5:
            action = 0;
            break;
        }
        takeAction(action);
    }

    public void qLearn() {
        totalReward += RL.reward;
        if (sLearn) {
            currentState = getState(distance);
            action = RL.selectAction(currentState);
            takeAction(action);
            sLearn = false;
        } else {
            state = getState(distance);
            nextState = state;
            RL.getQ(currentState, nextState, action);
            currentState = nextState;
            RL.reward = 0;
            action = RL.selectAction(currentState);
            takeAction(action);
        }
    }

    public void sarsa() {
        if (sLearn) {
            s3 = getState(distance);
            a3 = (int) (Math.random() * (numActions - 1));
            currentState = s3;
            RL.reward = 0;
            takeAction(a3);
            sLearn = false;
        } else if (sL) {
            r2 = RL.reward;

            s2 = s3;
            a2 = a3;
            s3 = getState(distance);

            a3 = (int) (Math.random() * (numActions - 1));
            currentState = s3;
            RL.reward = 0;
            takeAction(a3);
            sL = false;
        } else {
            state = getState(distance);
            if (state != s3) {
                r1 = r2;
                r2 = RL.reward;
                s1 = s2;
                a1 = a2;
                s2 = s3;
                a2 = a3;
                s3 = state;
                currentState = s3;

                RL.getSarsa(s1, a1, s2, a2, r1);
                RL.reward = 0;
                a3 = RL.selectAction(s1);
            }
            takeAction(a3);
        }
    }

    @Override
    public void onScannedRobot(ScannedRobotEvent e) {

        robotEnergy = getEnergy();
        enemyEnergy = e.getEnergy();
        bearing = e.getBearing();
        heading = e.getHeading();
        distance = e.getDistance();

        qLearn();
        // sarsa();
    }

    @Override
    public void onHitRobot(HitRobotEvent e) {
        bearing = e.getBearing();
        heading = e.getBearing();
    }

    public void onHitByBullet(HitByBulletEvent e) {
        bearing = e.getBearing();
        heading = e.getHeading();
        // hardCode();
        qLearn();
        // sarsa();
    }

    @Override
    public void onBulletHit(BulletHitEvent e) {
        // hardCode();
        qLearn();
        // sarsa();
    }

    @Override
    public void onDeath(DeathEvent e) {
        RL.reward -= 10.0;
        totalReward += RL.reward;
    }

    @Override
    public void onWin(WinEvent event) {
        RL.reward += 10.0;
        totalReward += RL.reward;
        // saveData();

        wins++;
    }


    public int getState(double distance) {
        int distState = 0;
        int enerState = 0;

        if (distance < 50) {
            distState = 0;
        } else if (distance < 100) {
            distState = 1;
        } else {
            distState = 2;
        }

        if (getEnergy() < 70) {
            enerState = 0;
        } else {
            enerState = 1;
        }

        state = RL.states[distState][enerState];

        return state;
    }

    public void target(int power) {
        if (power == 3) {
            if (getEnergy() > 70 && distance > 50) {
                RL.reward += 5.0;
            } else {
                RL.reward -= 5.0;
            }
        } else if (power == 1) {
            if (getEnergy() < 70 && distance > 50) {
                RL.reward += 5.0;
            } else {
                RL.reward -= 5.0;
            }
        }

        double turn = getHeading() - getGunHeading() + bearing;
        turnGunRight(normalise(turn));
        fire(power);
    }

    public void dodge() {
        if (distance < 50 && getEnergy() < 70) {
            RL.reward += 5.0;
        } else {
            RL.reward -= 5.0;
        }
        turnRight(Utils.normalRelativeAngleDegrees(90 - (getHeading() - heading)));

        ahead(150);
    }

    public void aggFire() {
        if (distance < 50 && getEnergy() > 50) {
            RL.reward += 5.0;
        } else {
            RL.reward -= 5.0;
        }
        double turnGunAmt = Utils.normalRelativeAngleDegrees(bearing + getHeading() - getGunHeading());

        turnGunRight(turnGunAmt);
        fire(3);
    }

    public double normalise(double angle) {

        while (angle > 180) {
            angle -= 360;
        }
        while (angle < -180) {
            angle += 360;
        }
        return angle;
    }

    @Override
    public void onRoundEnded(RoundEndedEvent e) {
        numRounds++;
        nR++;
        RL.epsilon -= 0.0016;

        if (numRounds % 25 == 0) {
            totR[count] = totalReward;
            count++;
        }
        totalReward = 0;

        numActions = 0;
        numCorrectActions = 0;
    }

    @Override
    public void onBattleEnded(BattleEndedEvent e) {
        saveData();
        l.QtoLUT();
        saveLUT();
        n.QtoNN();
        try {
            n.saveNN();
        } catch (IOException ex) {
            System.out.println("IOEXCEPTION");
        }
    }

    public void takeAction(int action) {
        numActions++;
        switch (action) {
        case 0:
            target(3);
            break;

        case 1:
            target(1);
            break;

        case 2:
            dodge();
            break;

        case 3:
            aggFire();
            break;
        }
    }

    public void saveLUT() {
        File file = new File("robProj/LUT");
        try {
            l.save(file);
        } catch (IOException e) {
            System.out.println("IOEXCEPTION");
        }
    }

    public void saveData() {
        PrintStream w = null;
        File file = getDataFile("r2.txt");

        try {
            w = new PrintStream(new RobocodeFileOutputStream(file));
            for (int i = 0; i < totR.length; i++) {
                if (!Double.isNaN(totR[i])) {
                    w.println(totR[i]);
                } else {
                    w.println(1.0);
                }
            }
        } catch (IOException e) {
            System.out.println("IOEXCEPTION");
        } finally {
            try {
                w.close();
            } catch (Exception e) {
                System.out.println("EXCEPTION: " + e);
            }
        }
        System.out.println("Saving done");
    }
}
