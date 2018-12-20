package com.zpj.dl4jtest;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;
import java.util.Map;
import java.util.Random;

public class DeepLearning4jTest {

    //随机数种子，用于结果复现
    private static final int seed = 12345;
    private static final Random random = new Random(seed);
    //随机数生成范围
    private static final int MIN_RANGE = 0;
    private static final int MAX_RANGE = 3;

    //网络模型学习率
    private static final double learningRate = 0.01;
    //对于每个miniBatch的迭代次数
    private static final int iterations = 10;
    //epoch数量(全部数据训练次数)
    private static final int nEpochs = 20;
    //生成样本点数量
    private static final int nSimples = 1000;
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    private static final int batchSize = 100;

    public static void main(String[] args) {

        int numInput = 1;
        int numOutput = 1;

        /**
         * 神经网络配置
         * 配置神经网络超参数
         */
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                /**
                 * 设置随机种子
                 */
                .seed(seed)
                /**
                 * 找方向
                 */
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                /**
                 * 对神经网络的权重进行随机初始化
                 */
                .weightInit(WeightInit.XAVIER)
                /**
                 * 优化算法
                 *
                 * 迈步子
                 */
                .updater(new Sgd(learningRate))
                .list()
                .layer(
                        0,
                        new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numInput) //上一层输入
                        .nOut(numOutput) //当前层神经单元的个数
                        .build()
                )
                .build();

        /**
         * 对神经网络进行构建
         */
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(configuration);
        /**
         * 必须调用init()
         * 对于模型参数的初始化
         */
        multiLayerNetwork.init();

        System.out.println(multiLayerNetwork.summary());




        DataSetIterator dataSetIterator = getTrainData(batchSize, random);

        //训练整个数据集nEpochs次
        for (int i = 0; i < nEpochs; i++) {
            dataSetIterator.reset();

            //用于训练模型
            multiLayerNetwork.fit(dataSetIterator);

            Map<String, INDArray> params = multiLayerNetwork.paramTable();
            params.forEach((key, value) -> System.out.println("key = " + key + "， value = " + value));
        }
        //测试两个数字，判断
        final INDArray input = Nd4j.create(new double[] { 10, 100 }, new int[] { 2, 1 });
        INDArray out = multiLayerNetwork.output(input, false);
        System.out.println(out);
    }

    private static DataSetIterator getTrainData(int batchSize, Random random){
        /**
         * 如何构造训练数据
         * 现有模型主要是有监督学习
         * 我们的训练集必须有 特征+标签
         * 特征-> x
         * 标签-> y
         */
        double[] output = new double[nSimples];
        double[] input = new double[nSimples];

        for (int i = 0; i < nSimples; i++) {
            input[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * random.nextDouble();
            output[i] = 2 * input[i] + 3;
        }

        /**
         * nSimples条数据
         * 每条数据只有1个x
         */
        INDArray inputArray = Nd4j.create(input, new int[]{nSimples, 1});
        INDArray outputArray = Nd4j.create(output, new int[]{nSimples, 1});

        /**
         * 构造喂给神经网路的数据集
         * DataSet是将 特征+标签 包装成为一个类
         */
        DataSet dataSet = new DataSet(inputArray, outputArray);
        List<DataSet> dataSetList = dataSet.asList();

        return new ListDataSetIterator<DataSet>(dataSetList, batchSize);
    }
}
