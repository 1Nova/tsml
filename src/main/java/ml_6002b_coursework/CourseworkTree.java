package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import java.util.Arrays;
import java.util.Random;

/**
 * A basic decision tree classifier for use in machine learning coursework (6002B).
 */
public class CourseworkTree extends AbstractClassifier {

    /** Measure to use when selecting an attribute to split the data with. */
    private AttributeSplitMeasure attSplitMeasure = new IGAttributeSplitMeasure();

    /** Maxiumum depth for the tree. */
    private int maxDepth = Integer.MAX_VALUE;

    /** The root node of the tree. */
    private TreeNode root;

    //function to set options:
    //we'll simply use the later defined setAttSplitMeasure() method
    @Override
    public void setOptions(String[] options) {
        for (String option: options){
            //cast to lower case: we don't want to handle lots of different input variations!
            switch (option.toLowerCase()) {
                case "gain":
                    //use constructor to set gain args, via useGain boolean
                    setAttSplitMeasure(new IGAttributeSplitMeasure(true));
                    break;
                case "ratio":
                    //use constructor to set ratio args, via useGain boolean
                    setAttSplitMeasure(new IGAttributeSplitMeasure(false));
                    break;
                case "chi":
                    setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
                case "gini":
                    setAttSplitMeasure(new GiniAttributeSplitMeasure());
                    break;
                default:
                    break;
            }
        }
    }


    /**
     * Sets the attribute split measure for the classifier.
     *
     * @param attSplitMeasure the split measure
     */
    public void setAttSplitMeasure(AttributeSplitMeasure attSplitMeasure) {
        this.attSplitMeasure = attSplitMeasure;
    }

    /**
     * Sets the max depth for the classifier.
     *
     * @param maxDepth the max depth
     */
    public void setMaxDepth(int maxDepth){
        this.maxDepth = maxDepth;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        //instances
        result.setMinimumNumberInstances(2);

        return result;
    }

    /**
     * Builds a decision tree classifier.
     *
     * @param data the training data
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes() - 1) {
            throw new Exception("Class attribute must be the last index.");
        }

        root = new TreeNode();
        root.buildTree(data, 0);
    }

    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the classification
     */
    @Override
    public double classifyInstance(Instance instance) {
        double[] probs = distributionForInstance(instance);

        int maxClass = 0;
        for (int n = 1; n < probs.length; n++) {
            if (probs[n] > probs[maxClass]) {
                maxClass = n;
            }
        }

        return maxClass;
    }

    /**
     * Computes class distribution for instance using the decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     */
    @Override
    public double[] distributionForInstance(Instance instance) {
        return root.distributionForInstance(instance);
    }

    /**
     * Class representing a single node in the tree.
     */
    private class TreeNode {

        /** Attribute used for splitting, if null the node is a leaf. */
        Attribute bestSplit = null;

        /** Best gain from the splitting measure if the node is not a leaf. */
        double bestGain = 0;

        /** Depth of the node in the tree. */
        int depth;

        /** The node's children if it is not a leaf. */
        TreeNode[] children;

        /** The class distribution if the node is a leaf. */
        double[] leafDistribution;

        /**
         * Recursive function for building the tree.
         * Builds a single tree node, finding the best attribute to split on using a splitting measure.
         * Splits the best attribute into multiple child tree node's if they can be made, else creates a leaf node.
         *
         * @param data Instances to build the tree node with
         * @param depth the depth of the node in the tree
         */
        void buildTree(Instances data, int depth) throws Exception {
            this.depth = depth;

            // Loop through each attribute, finding the best one.
            for (int i = 0; i < data.numAttributes() - 1; i++) {
                double gain;

                gain = attSplitMeasure.computeAttributeQuality(data, data.attribute(i));

                //System.out.println("gain:" + gain);

                if (gain > bestGain) {
                    bestSplit = data.attribute(i);
                    bestGain = gain;
                }
            }

            //System.out.println(bestSplit);


            // If we found an attribute to split on, create child nodes.
            if (bestSplit != null) {
                Instances[] split = new Instances[0];

                //if type numeric, use on numeric method
                //todo maybe use .type() == 0   OR use .isNumeric()
                if (bestSplit.isNumeric()){
                    split = attSplitMeasure.splitDataOnNumeric(data, bestSplit);
                }
                //if nominal, use normal method
                else if (bestSplit.isNominal()){
                    split = attSplitMeasure.splitData(data, bestSplit);
                }
                children = new TreeNode[split.length];

                // Create a child for each value in the selected attribute, and determine whether it is a leaf or not.
                for (int i = 0; i < children.length; i++){
                    children[i] = new TreeNode();

                    boolean leaf = split[i].numDistinctValues(data.classIndex()) == 1 || depth + 1 == maxDepth;

                    if (split[i].isEmpty()) {
                        children[i].buildLeaf(data, depth + 1);
                    } else if (leaf) {
                        children[i].buildLeaf(split[i], depth + 1);
                    } else {
                        children[i].buildTree(split[i], depth + 1);
                    }
                }
            // Else turn this node into a leaf node.
            } else {
                leafDistribution = classDistribution(data);
            }
        }

        /**
         * Builds a leaf node for the tree, setting the depth and recording the class distribution of the remaining
         * instances.
         *
         * @param data remaining Instances to build the leafs class distribution
         * @param depth the depth of the node in the tree
         */
        void buildLeaf(Instances data, int depth) {
            this.depth = depth;
            leafDistribution = classDistribution(data);
        }

        /**
         * Recursive function traversing node's of the tree until a leaf is found. Returns the leafs class distribution.
         *
         * @return the class distribution of the first leaf node
         */
        double[] distributionForInstance(Instance inst) {
            // If the node is a leaf return the distribution, else select the next node based on the best attributes
            // value.
            if (bestSplit == null) {
                //System.out.println("bestsplit null");
                return leafDistribution;
            } else {
                //if numeric
                if (bestSplit.isNumeric()){
                    //System.out.println(attSplitMeasure.getRandom());
                    //System.out.println("numeric cwtree 248");
                    if(inst.value(bestSplit) <= attSplitMeasure.getRandom()){
                        return children[1].distributionForInstance(inst);
                    }
                    //if nominal
                    else{
                        //System.out.println("nominal cwtree 254");
                        return children[0].distributionForInstance(inst);
                    }
                }
                return children[(int) inst.value(bestSplit)].distributionForInstance(inst);
            }
        }

        /**
         * Returns the normalised version of the input array with values summing to 1.
         *
         * @return the class distribution as an array
         */
        double[] classDistribution(Instances data) {
            double[] distribution = new double[data.numClasses()];
            for (Instance inst : data) {
                distribution[(int) inst.classValue()]++;
            }

            double sum = 0;
            for (double d : distribution){
                sum += d;
            }

            if (sum != 0){
                for (int i = 0; i < distribution.length; i++) {
                    distribution[i] = distribution[i] / sum;
                }
            }

            return distribution;
        }

        /**
         * Summarises the tree node into a String.
         *
         * @return the summarised node as a String
         */
        @Override
        public String toString() {
            String str;
            if (bestSplit == null){
                str = "Leaf," + Arrays.toString(leafDistribution) + "," + depth;
            } else {
                str = bestSplit.name() + "," + bestGain + "," + depth;
            }
            return str;
        }
    }

    /**
     * Main method.
     *
     * @param args the options for the classifier main
     */
    public static void main(String[] args) throws Exception {

        // Read all the instances in the file (ARFF, CSV, XRFF, ...)
        ConverterUtils.DataSource optdigitsSource =
                new ConverterUtils.DataSource("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");
        ConverterUtils.DataSource chinaSource =
                new ConverterUtils.DataSource("src/main/java/ml_6002b_coursework/test_data/Chinatown.arff");

        // Optdigits
        System.out.println("DT using measure 'Information Gain' on optdigits problem has test accuracy = " + accuracyMetrics(optdigitsSource, new IGAttributeSplitMeasure(true)));
        System.out.println("hi");
        //System.out.println("DT using measure 'Information Gain Ratio' on optdigits problem has test accuracy = " + accuracyMetrics(optdigitsSource, new IGAttributeSplitMeasure(false)));
        System.out.println("DT using measure 'Chi Squared' on optdigits problem has test accuracy = " + accuracyMetrics(optdigitsSource, new ChiSquaredAttributeSplitMeasure()));
        System.out.println("DT using measure 'Gini Index' on optdigits problem has test accuracy = " + accuracyMetrics(optdigitsSource, new GiniAttributeSplitMeasure()));

        // chinaTowns
        System.out.println("\nDT using measure 'Information Gain' on Chinatown problem has test accuracy = " + accuracyMetrics(chinaSource, new IGAttributeSplitMeasure(true)));
        System.out.println("DT using measure 'Information Gain Ratio' on Chinatown problem has test accuracy = " + accuracyMetrics(chinaSource, new IGAttributeSplitMeasure(false)));
        System.out.println("DT using measure 'Chi Squared' on Chinatown problem has test accuracy = " + accuracyMetrics(chinaSource, new ChiSquaredAttributeSplitMeasure()));
        System.out.println("DT using measure 'Gini Index' on Chinatown problem has test accuracy = " + accuracyMetrics(chinaSource, new GiniAttributeSplitMeasure()));

    }

    public static double accuracyMetrics(ConverterUtils.DataSource source, AttributeSplitMeasure asm) throws Exception {

        Instances instances = source.getDataSet();

        // Make the last attribute be the class
        instances.setClassIndex(instances.numAttributes() - 1);


        //randomly split data
        StratifiedRemoveFolds filter = new StratifiedRemoveFolds();

        // -V
        //  Specifies if inverse of selection is to be output.
        //
        // -N <number of folds>
        //  Specifies number of folds dataset is split into.
        //  (default 10)
        //
        // -F <fold>
        //  Specifies which fold is selected. (default 1)
        //
        // -S <seed>
        //  Specifies random number seed. (default 0, no randomizing)
        String[] options = new String[6];
        Random rand = new Random();
        int seed = rand.nextInt(5000)+1;

        options[0] = "-N"; //number of folds (default is 10)
        options[1] = Integer.toString(5);
        options[2] = "-S"; //random number seed (default = 0, no randomisation)
        options[3] = Integer.toString(seed);
        options[4] = "-F"; //selected fold
        options[5] = Integer.toString(1);

        filter.setOptions(options); //apply options defined above
        filter.setInputFormat(instances); //define instances to be used
        filter.setInvertSelection(false);

        //filter for test data
        Instances testData = Filter.useFilter(instances, filter);

        //apply filter for training data instead now
        filter.setInvertSelection(true);
        Instances trainData = Filter.useFilter(instances, filter);

        CourseworkTree tree = new CourseworkTree();

        tree.setAttSplitMeasure(asm);

        //build using training data
        tree.buildClassifier(trainData);

        double correct = 0.0, total = 0.0;
        //test using test data
        for (Instance instance:testData){
            double prediction = tree.classifyInstance(instance);
            if (instance.classValue() == prediction){
                correct+=1;
            }
            total+=1;
        }
        double accuracy = correct/total;
        return accuracy;
    }
}