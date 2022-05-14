package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.stream.DoubleStream;

/**
 * Interface for alternative attribute split measures for Part 2.2 of the coursework
 */
public abstract class AttributeSplitMeasure extends AttributeMeasures {

    public abstract double computeAttributeQuality(Instances data, Attribute att) throws Exception;

    /**
     * Splits a dataset according to the values of a nominal attribute.
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
    public Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int i = 0; i < att.numValues(); i++) {
            splitData[i] = new Instances(data, data.numInstances());
        }

        for (Instance inst: data) {
            splitData[(int) inst.value(att)].add(inst);
        }

        for (Instances split : splitData) {
            split.compactify();
        }

        return splitData;
    }

    //Instantiation and getter for distributionForInstance() method in CourseworkTree
    double random;

    public double getRandom(){
        return this.random;
    }

    //generate split value (random)
    public double generateRandomSplitValue(Instances data, Attribute att) {
        AttributeStats attStats = data.attributeStats(att.index());
        double max = attStats.numericStats.max;
        double min = attStats.numericStats.min;
        double random = ((Math.random() * (max - min)) + min);
        return random;
    }

    public Instances[] splitDataOnNumeric(Instances data, Attribute att) {
        //create empty bin
        Instances[] splitData = new Instances[2];
        splitData[0] = new Instances(data, 0);
        splitData[1] = new Instances(data, 0);

        //use function above to generate random split value
        double random = generateRandomSplitValue(data, att);

        double mean = DoubleStream.of(data.attributeToDoubleArray(att.index())).sum() /
                data.attributeToDoubleArray(att.index()).length;

        //split instances into empty bin based on the generated split value
        for(int i = 0; i < data.numInstances(); i++)
        {
            Instance instance = data.instance(i);
            if (instance.value(att) <= random) {
                splitData[0].add(instance);
            }
            else {
                splitData[1].add(instance);
            }
        }

        //compactify!
        for (Instances ins : splitData) {
            ins.compactify();
        }

        return splitData;
    }

}
