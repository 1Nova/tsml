package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Arrays;


public class IGAttributeSplitMeasure extends AttributeSplitMeasure {

    boolean useGain;

    public IGAttributeSplitMeasure(boolean useGain) {
        this.useGain = useGain;
    }

    //note: empty constructor for "AttributeSplitMeasure" in CourseworkTree.java
    public IGAttributeSplitMeasure() {

    }

    public boolean isUseGain() {
        return useGain;
    }

    public void setUseGain(boolean useGain) {
        this.useGain = useGain;
    }

    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
//
//        Instances[] instArr;
//
//        if (att.isNumeric()){
//            instArr = splitDataOnNumeric(data, att);
//        }
//        else {
//            instArr = splitData(data, att);
//        }
////
//        int[][] arr = new int[att.numValues()][data.numClasses()];
//        for (int i = 0; i < instArr.length; i++){
//            for (int j = 0; j < instArr[i].size(); j++){
//                arr[i][(int) instArr[i].instance(j).classValue()]++;
//
//            }
//        }
//
//        for(Instance instance:splitData[i]){
//            value = (int)instance.classValue();
//            contingencyTable[i][value]++;
//
//        //System.out.println(Arrays.toString(instArr));
//
//        double quality = 0.0;
//
//        //initialise array
//        int[][] attributeArray = {
//                {0,    0},
//                {0,    0}
//        };
//
//        //update array
////        for(Instances insts : instArr) {
////            for (Instance instance : insts) {
////                // if islay
////                if (instance.value(3) == 1.0) {
////                    // if positive
////                    if (instance.value(att.index()) == 1.0) {
////                        attributeArray[0][0] = attributeArray[0][0] + 1;
////                    } else {
////                        attributeArray[1][0] = attributeArray[1][0] + 1;
////                    }
////                } /* else speyside */ else {
////                    // if positive
////                    if (instance.value(att.index()) == 1.0) {
////                        attributeArray[0][1] = attributeArray[0][1] + 1;
////                    } else {
////                        attributeArray[1][1] = attributeArray[1][1] + 1;
////                    }
////                }
////            }
////        }
//
////        System.out.println("new (splitting): " + Arrays.deepToString(arr));
////        System.out.println("old: " + Arrays.deepToString(attributeArray) + "\n");
//
//        if(useGain) {
//            quality = measureInformationGain(arr);
//        } else {
//            quality = measureInformationGainRatio(arr);
//        }
//
//        return quality;

        int count = data.numClasses();
        int value = att.numValues();
        System.out.println("test");
        if(att.isNumeric()){
            Instances[] splitData = splitDataOnNumeric(data,att);
            int[][] contingencyTable = new int[2][count];
            for (int i=0; i<2;i++){
                for(Instance instance:splitData[i]){
                    value = (int)instance.classValue();
                    contingencyTable[i][value]++;
                }
            }
            System.out.println(Arrays.deepToString(contingencyTable));
            if (useGain){
                return AttributeMeasures.measureInformationGainRatio(contingencyTable);
            } else {
                return AttributeMeasures.measureInformationGain(contingencyTable);
            }
        }else{
            int[][] contingencyTable = new int[value][count];

            for (Instance instance : data){
                int attributeValue = (int) instance.value(att);
                int classValue = (int) instance.classValue();
                contingencyTable[attributeValue][classValue]++;
            }
            if (useGain){
                return AttributeMeasures.measureInformationGainRatio(contingencyTable);
            } else {
                return AttributeMeasures.measureInformationGain(contingencyTable);
            }
        }
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

        IGAttributeSplitMeasure peatyGain = new IGAttributeSplitMeasure(true);
        IGAttributeSplitMeasure woodyGain = new IGAttributeSplitMeasure(true);
        IGAttributeSplitMeasure sweetGain = new IGAttributeSplitMeasure(true);
        System.out.println("measure 'Information Gain' for attribute 'peaty' splitting diagnosis = "
                + peatyGain.computeAttributeQuality(instances, peatyAtt));
        System.out.println("measure 'Information Gain' for attribute 'woody' splitting diagnosis = "
                + woodyGain.computeAttributeQuality(instances, woodyAtt));
        System.out.println("measure 'Information Gain' for attribute 'sweet' splitting diagnosis = "
                + sweetGain.computeAttributeQuality(instances, sweetAtt));


        IGAttributeSplitMeasure peatyRatio = new IGAttributeSplitMeasure(false);
        IGAttributeSplitMeasure woodyRatio = new IGAttributeSplitMeasure(false);
        IGAttributeSplitMeasure sweetRatio = new IGAttributeSplitMeasure(false);
        System.out.println("measure 'Information Gain Ratio' for attribute 'peaty' splitting diagnosis = "
                + peatyRatio.computeAttributeQuality(instances, peatyAtt));
        System.out.println("measure 'Information Gain Ratio' for attribute 'woody' splitting diagnosis = "
                + woodyRatio.computeAttributeQuality(instances, woodyAtt));
        System.out.println("measure 'Information Gain Ratio' for attribute 'sweet' splitting diagnosis = "
                + sweetRatio.computeAttributeQuality(instances, sweetAtt));

        //measure <insert> for attribute <insert> splitting diagnosis = <insert>

    }

}
