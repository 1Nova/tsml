package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Arrays;


public class GiniAttributeSplitMeasure extends AttributeSplitMeasure {

    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        double quality = 0.0;

        //initialise array
        int[][] attributeArray = {
                {0,    0},
                {0,    0}
        };

        //update array
        for(Instance instance : data) {
            // if islay
            if(instance.value(3) == 1.0) {
                // if positive
                if(instance.value(att.index()) == 1.0) {
                    attributeArray[0][0] = attributeArray[0][0] + 1;
                } else {
                    attributeArray[1][0] = attributeArray[1][0] + 1;
                }
            } /* else speyside */
            else {
                // if positive
                if(instance.value(att.index()) == 1.0) {
                    attributeArray[0][1] = attributeArray[0][1] + 1;
                } else {
                    attributeArray[1][1] = attributeArray[1][1] + 1;
                }
            }
        }

        quality = measureGini(attributeArray);

        return quality;
    }

    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) throws Exception {

        // Read all the instances in the file (ARFF, CSV, XRFF, ...)
        DataSource source = new DataSource("src/main/java/ml_6002b_coursework/test_data/whiskey.arff");
        Instances instances = source.getDataSet();

        // Make the last attribute be the class
        instances.setClassIndex(instances.numAttributes() - 1);

        Attribute peatyAtt = instances.attribute("peaty");
        Attribute woodyAtt = instances.attribute("woody");
        Attribute sweetAtt = instances.attribute("sweet");

        GiniAttributeSplitMeasure peatyGini = new GiniAttributeSplitMeasure();
        GiniAttributeSplitMeasure woodyGini = new GiniAttributeSplitMeasure();
        GiniAttributeSplitMeasure sweetGini = new GiniAttributeSplitMeasure();
        System.out.println("measure 'Gini index' for attribute 'peaty' splitting diagnosis = "
                + peatyGini.computeAttributeQuality(instances, peatyAtt));
        System.out.println("measure 'Gini index' for attribute 'woody' splitting diagnosis = "
                + woodyGini.computeAttributeQuality(instances, woodyAtt));
        System.out.println("measure 'Gini index' for attribute 'sweet' splitting diagnosis = "
                + sweetGini.computeAttributeQuality(instances, sweetAtt));

    }

}
