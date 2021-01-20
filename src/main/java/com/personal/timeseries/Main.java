package com.personal.timeseries;

import java.io.File;
import java.io.IOException;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;


import org.nd4j.evaluation.classification.ROC;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Main {

    public static void main(String[] args) throws IOException, InterruptedException {
        String featureFolder = "/home/luis/NetBeansProjects/TimeSeries/Data/physionet2012/sequence";
        String mortalityFolder = "/home/luis/NetBeansProjects/TimeSeries/Data/physionet2012/mortality";
        int random = 1234;
        
        SequenceRecordReader trainFeaturesReader = new CSVSequenceRecordReader(1,",");
        trainFeaturesReader.initialize(new NumberedFileInputSplit(featureFolder+"/%d.csv",0,3199));
        SequenceRecordReader trainLabelReader = new CSVSequenceRecordReader();
        trainLabelReader.initialize(new NumberedFileInputSplit(mortalityFolder+"/%d.csv",0,3199));
        DataSetIterator trainDataSetIterator = new SequenceRecordReaderDataSetIterator(trainFeaturesReader, trainLabelReader, 100,2,false,SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        SequenceRecordReader testFeaturesReader = new CSVSequenceRecordReader(1,",");
        testFeaturesReader.initialize(new NumberedFileInputSplit(featureFolder+"/%d.csv",3200,3999));
        SequenceRecordReader testLabelReader = new CSVSequenceRecordReader();
        testLabelReader.initialize(new NumberedFileInputSplit(mortalityFolder+"/%d.csv",3200,3999));
        DataSetIterator testDataSetIterator = new SequenceRecordReaderDataSetIterator(testFeaturesReader, testLabelReader, 100,2,false,SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);        
        
        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
                                                .seed(random)
                                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                                .weightInit(WeightInit.XAVIER)
                                                .updater(new Adam())
                                                .dropOut(0.9)
                                                .graphBuilder()
                                                .addInputs("trainFeatures")
                                                .setOutputs("predictMortality")
                                                .addLayer("L1", new LSTM.Builder()
                                                                               .nIn(86)
                                                                               .nOut(200)
                                                                               .forgetGateBiasInit(1)
                                                                               .activation(Activation.TANH)
                                                                               .build(),"trainFeatures")
                                                .addLayer("predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                                                    .activation(Activation.SOFTMAX)
                                                                                    .nIn(200).nOut(2).build(),"L1")
                                                .build();
        


        ComputationGraph model = new ComputationGraph(configuration);

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage(); 
        model.setListeners(new StatsListener(statsStorage));     
        uiServer.attach(statsStorage);

//        for(int i=0;i<5;i++){
//           model.fit(trainDataSetIterator);
//           trainDataSetIterator.reset();
//        }

        model.fit(trainDataSetIterator,1);
        ROC evaluation = new ROC(100);
        while (testDataSetIterator.hasNext()) {
            DataSet batch = testDataSetIterator.next();
            INDArray[] output = model.output(batch.getFeatures());
            evaluation.evalTimeSeries(batch.getLabels(), output[0]);
        }
        
        System.out.println(evaluation.calculateAUC());
        System.out.println(evaluation.stats());
        
        File file = new File("model.zip");
        ModelSerializer.writeModel(model, file, true);
        
    }    
}
