package ml_6002b_coursework;

import java.util.ArrayList;
import java.util.Arrays;

import static weka.core.Utils.log2;

/**
 * Empty class for Part 2.1 of the coursework.
 */
public class AttributeMeasures {

    /**
     * Main method.
     *
     * @param args the options for the attribute measure main
     */
    public static void main(String[] args) throws Exception {
        //for final column, islay = 1, speyside = 0
        //public int[][] testarray = new int[][];
        int[][] testArray =
                {{1,0,1,1},
                {1,1,1,1},
                {1,0,0,1},
                {1,0,0,1},
                {0,1,0,1},
                {0,1,1,0},
                {0,1,1,0},
                {0,1,1,0},
                {0,0,1,0},
                {0,0,1,0},};

        int[][] peatyArray = {
                //              Islay, Speyside
                //              1,     0
                /*Peaty*/       {4,    0}, //4
                /*Not Peaty*/   {1,    5}  //6
        };                     //5     5     10

        System.out.println("measure information gain for Peaty = " + measureInformationGain(peatyArray));
        System.out.println("measure information gain ratio for Peaty = " + measureInformationGainRatio(peatyArray));
        System.out.println("measure gini for Peaty = " + measureGini(peatyArray));
        System.out.println("measure chi squared for Peaty = " + measureChiSquared(peatyArray));

    }

    //entropy method: pass in two doubles of (prob1, prob2)
    static double entropy(double class0, double class1) {
            return 0-(class0 * log2(class0) + class1 * log2(class1));
    }

    // Rows = different values of the attribute being assessed
    // Columns = class counts
    public static double measureInformationGain(int[][] arrayArgs) {

        //total of all classes: will also be used for our IG calculation as we need to use global probability
        double parentTotal = 0.0;

        ArrayList<Integer> parentClasses = new ArrayList<>();

        //iterate columns, summing
        for(int i = 0; i < arrayArgs[0].length; i++){
            int columnSum = 0;
            for(int j = 0; j < arrayArgs.length; j++){
                columnSum = columnSum + arrayArgs[j][i];
            }
            parentClasses.add(columnSum);
            parentTotal = parentTotal + columnSum;
        }

        double parentClass0 = parentClasses.get(0) / parentTotal;
        double parentClass1 = parentClasses.get(1) / parentTotal;

        double parentEntropy = entropy(parentClass0, parentClass1);

        ArrayList<Double> weightedEntropies = new ArrayList<>();

        //iterate rows
        for(int i=0; i < arrayArgs.length; i++) {
            double rowTotal = Arrays.stream(arrayArgs[i]).sum();

            //todo note: assuming first column is our desired 'positive', ie. in this case islay
            //  -> may have to make it a method so we can pass in desired column 'i' instead of default [0]
            double class0 = arrayArgs[i][0] / rowTotal;
            double class1 = (rowTotal - arrayArgs[i][0]) / rowTotal;
            double entropyRow = entropy(class0, class1);
            weightedEntropies.add((rowTotal/parentTotal)*entropyRow);

        }


        //sum weighted entropies for IG calculation, ignoring 'NaN' (not a number) types
        double sumWeightedEntropies = 0;
        for(Double d : weightedEntropies) {
            if (!Double.isNaN(d)) {
                sumWeightedEntropies += d;
            }
        }

        double informationGain = parentEntropy - (sumWeightedEntropies);
        return informationGain;
    }

    public static double measureInformationGainRatio(int[][] arrayArgs) {

        int rows = arrayArgs.length;
        int cols = arrayArgs[0].length;

        ArrayList<Integer> rowsArray = new ArrayList<>();
        double parentTotal = 0.0;
        for(int i = 0; i < rows; i++){
            int rowSum = 0;
            for(int j = 0; j < cols; j++){
                rowSum = rowSum + arrayArgs[i][j];
            }
            rowsArray.add(rowSum);
            parentTotal = parentTotal + rowSum;
        }

//        for( int i=0; i < rowsArray.size() ;i++) {
//            // Difference = next record - current record
//            rowsArray[i]
//        }

        double splitInformation = 0.0;
        for(int row : rowsArray) {
            //double rowSplitInformation = -(row/parentTotal)*log2(row/parentTotal);
            splitInformation = splitInformation - ((row/parentTotal)*log2(row/parentTotal));
        }
        double informationGainRatio = measureInformationGain(arrayArgs)/splitInformation;
        return informationGainRatio;
    }

    public static double measureGini(int[][] arrayArgs) {

        //PARENT GINI
        //total of all classes: will also be used for our IG calculation as we need to use global probability
        double parentTotal = 0.0;

        ArrayList<Integer> parentClasses = new ArrayList<>();

        //iterate columns, summing
        for(int i = 0; i < arrayArgs[0].length; i++){
            int columnSum = 0;
            for(int j = 0; j < arrayArgs.length; j++){
                columnSum = columnSum + arrayArgs[j][i];
            }
            parentClasses.add(columnSum);
            parentTotal = parentTotal + columnSum;
        }

        ArrayList<Double> ginis = new ArrayList<>();

        //iterate rows
        for(int i=0; i < arrayArgs.length; i++) {
            double rowTotal = Arrays.stream(arrayArgs[i]).sum();

            //todo note: assuming first column is our desired 'positive', ie. in this case islay
            //  -> may have to make it a method so we can pass in desired column 'i' instead of default [0]
            double class0 = arrayArgs[i][0] / rowTotal;
            double class1 = (rowTotal - arrayArgs[i][0]) / rowTotal;

            double calc = class0;
            double calc2 = class1;
            ginis.add((Math.pow(calc, 2)*(rowTotal/parentTotal) + Math.pow(calc2, 2)*(rowTotal/parentTotal)));
        }

        double parentClass0 = parentClasses.get(0) / parentTotal; //5 / 10
        double parentClass1 = parentClasses.get(1) / parentTotal; // 5 / 10
        double parentGini = 1 - (Math.pow(parentClass0, 2) + Math.pow(parentClass1, 2));

        ArrayList<Double> weightedEntropies = new ArrayList<>();

        double gini = parentGini - (ginis.get(0) + ginis.get(1));
        return -gini;
    }

    public static double measureChiSquared(int[][] arrayArgs) {
        double globalTotal = 0.0;
        ArrayList<Integer> columnSums = new ArrayList<>();

        //loop to get sums for probabilities
        //iterate columns, summing
        for(int i = 0; i < arrayArgs[0].length; i++){
            int columnSum = 0;
            for(int j = 0; j < arrayArgs.length; j++){
                columnSum = columnSum + arrayArgs[j][i];
            }
            globalTotal = globalTotal + columnSum;
            columnSums.add(columnSum);
        }

        ArrayList<Double> chis = new ArrayList<>();

        //iterate rows
        for(int i=0; i < arrayArgs.length; i++) {

            double rowTotal = Arrays.stream(arrayArgs[i]).sum();

            double expectedValue0 = (rowTotal * (columnSums.get(0)/globalTotal));
            double expectedValue1 = (rowTotal * (columnSums.get(1)/globalTotal));
            double observedValue0 = arrayArgs[i][0];
            double observedValue1 = arrayArgs[i][1];

            double class0 = Math.pow((observedValue0 - expectedValue0), 2) / expectedValue0;
            double class1 = Math.pow((observedValue1 - expectedValue1), 2) / expectedValue1;

            chis.add(class0 + class1);

            double rowProbability = rowTotal/columnSums.get(i);

        }
        double chiSquared = 0.0;
        for(double chi : chis) {
            chiSquared = chiSquared + chi;
        }

        return chiSquared;
    }
}
