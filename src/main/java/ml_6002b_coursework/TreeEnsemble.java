package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.ArrayList;
import java.util.Random;

public class TreeEnsemble extends AbstractClassifier {
    int numTrees;

    double attributeProportion;

    int attributeIndex;

    ArrayList<CourseworkTree> classifiers = new ArrayList<>();

    int[] newAttributes;


    ArrayList<Integer> attributesList = new ArrayList<>();

    public TreeEnsemble() {
        //default values
        this.numTrees = 50;
        this.attributeProportion = 0.5;
    }

    public void setNumTrees(int numTrees) {
        this.numTrees = numTrees;
    }

    public void setAttributeProportion(double attributeProportion) {
        this.attributeProportion = attributeProportion;
    }


    @Override
    public void buildClassifier(Instances data) throws Exception {

        int[] newAttributes = new int[(int)(data.numAttributes() * attributeProportion)];




        Random rand = new Random();

        for (int i = 0; i < numTrees; i++) {
            for (int y =0; y < data.numAttributes()-1; y++){
                attributesList.add(y);
            }

            for (int x = 0; x < (data.numAttributes() * attributeProportion); x++){
                attributeIndex = rand.nextInt(attributesList.size()-1);
                newAttributes[x] = attributesList.get(attributeIndex);
                attributesList.remove(attributeIndex);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource optdigitsSource =
                new ConverterUtils.DataSource("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");

        Instances data = optdigitsSource.getDataSet();

        TreeEnsemble te = new TreeEnsemble();
        te.buildClassifier(data);
        System.out.println(te.attributesList);
        //System.out.println(//newAttributes);
    }
}
