package com.zpj.dl4jtest;

import com.zpj.dl4jtest.util.LogUtil;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class MnistTest {

    public static void main(String[] args) throws Exception {

        final int numRows = 28; //矩阵的行数
        final int numColumns = 28; //矩阵的列数
        int outputNum = 10; //潜在结果（比如0到9的整数标签）的数量
        int batchSize = 128; //每一步抓取的样例数量
        int seed = 123; //随机数生成器用一个随机种子来确保定型时使用的初始权重维持一样
        int numEpochs = 15; //一个epoch指将给定数据集来全部处理一遍的周期

        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        LogUtil.d("", "Build model...");
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4)
                .list()
                .layer(
                        0,
                        new DenseLayer.Builder()
                                .nIn(numRows * numColumns)
                                .nOut(1000)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER)
                                .build()
                )
                .layer(
                        1,
                        new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(1000)
                                .nOut(outputNum)
                                .activation(Activation.SOFTMAX)
                                .weightInit(WeightInit.XAVIER)
                                .build()
                )
                .build();

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(configuration);
        multiLayerNetwork.init();

        multiLayerNetwork.setListeners(new ScoreIterationListener(1));

        LogUtil.d("", "Train model...");
        for (int i = 0; i < numEpochs; i++) {
            multiLayerNetwork.fit(mnistTrain);
        }

        LogUtil.d("", "Evaluate model...");
        Evaluation eval = new Evaluation(outputNum);
        while (mnistTest.hasNext()) {
            DataSet next = mnistTest.next();
            INDArray output = multiLayerNetwork.output(next.getFeatures());
            eval.eval(next.getLabels(), output);
        }

        LogUtil.d("", eval.stats());
        LogUtil.d("", "finished..........................");
    }
}
