import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.Scanner;

/**
 * Created by PT3 on 3/8/2018.
 */

public class RandomizedOptimization {

    private static Instance[][] instances = initializeInstances();

    private static Instance[] trainInstances = instances[0];
    private static Instance[] testInstances = instances[1];

    private static int inputLayer = 36, hiddenLayer = 15, outputLayer = 1, trainingIterations = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(trainInstances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < testInstances.length; j++) {
                networks[i].setInputValues(testInstances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(testInstances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < trainInstances.length; j++) {
                network.setInputValues(trainInstances[j].getData());
                network.run();

                Instance output = trainInstances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            //System.out.println(df.format(error));
        }
    }

    private static Instance[][] initializeInstances() {

        double[][][] trainAttributes = new double[2500][][];

        double[][][] testAttributes = new double[695][][];

        try{
            BufferedReader br = new BufferedReader(new FileReader(new File("..//chessdata.csv")));



            for(int i = 0; i < trainAttributes.length; i++) {

                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                trainAttributes[i] = new double[2][];
                trainAttributes[i][0] = new double[36];
                trainAttributes[i][1] = new double[1];

                for(int j = 0; j < 36; j++)
                    trainAttributes[i][0][j] = Double.parseDouble(scan.next());

                String label = scan.next();
                if (label.equals("won")) {
                    trainAttributes[i][1][0] = 1;
                } else {
                    trainAttributes[i][1][0] = 0;
                }
            }

            for(int i = 0; i < testAttributes.length; i++) {

                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                testAttributes[i] = new double[2][];
                testAttributes[i][0] = new double[36];
                testAttributes[i][1] = new double[1];

                for(int j = 0; j < 36; j++)
                    testAttributes[i][0][j] = Double.parseDouble(scan.next());

                String label = scan.next();
                if (label.equals("won")) {
                    testAttributes[i][1][0] = 1;
                } else {
                    testAttributes[i][1][0] = 0;
                }
            }


        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] trainInstances = new Instance[trainAttributes.length];

        Instance[] testInstances = new Instance[testAttributes.length];

        for(int i = 0; i < trainInstances.length; i++) {
            trainInstances[i] = new Instance((trainAttributes[i][0]));
            trainInstances[i].setLabel(new Instance(trainAttributes[i][1][0]));
        }

        for(int i = 0; i < testInstances.length; i++) {
            testInstances[i] = new Instance((testAttributes[i][0]));
            testInstances[i].setLabel(new Instance(testAttributes[i][1][0]));
        }

        Instance[][] instances = {trainInstances, testInstances};

        return instances;



    }

}
