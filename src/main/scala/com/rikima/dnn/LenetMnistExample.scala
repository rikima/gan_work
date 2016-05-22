package com.rikima.dnn

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
//import org.deeplearning4j.nn.conf.LearningRatePolicy
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * Created by a14350 on 2016/05/22.
  */
object LenetMnistExample {


  def main(args: Array[String]) {
    val nChannels = 1
    val outputNum = 10
    val batchSize = 64
    val nEpochs = 10
    val iterations = 1
    val seed = 123

    println("Load data....")
    val mnistTrain = new MnistDataSetIterator(batchSize,true,12345)
    val mnistTest = new MnistDataSetIterator(batchSize,false,12345)

    println("Build model....")
    val builder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .regularization(true).l2(0.0005)
      .learningRate(0.01)//.biasLearningRate(0.02)
      //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .list()
      .layer(0, new ConvolutionLayer.Builder(5, 5)
        .nIn(nChannels)
        .stride(1, 1)
        .nOut(20)
        .activation("identity")
        .build())
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2,2)
        .stride(2,2)
        .build())
      .layer(2, new ConvolutionLayer.Builder(5, 5)
        .nIn(nChannels)
        .stride(1, 1)
        .nOut(50)
        .activation("identity")
        .build())
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2,2)
        .stride(2,2)
        .build())
      .layer(4, new DenseLayer.Builder().activation("relu")
        .nOut(500).build())
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(outputNum)
        .activation("softmax")
        .build())
      .backprop(true).pretrain(false)

    new ConvolutionLayerSetup(builder,28,28,1)

    val conf = builder.build()
    val model = new MultiLayerNetwork(conf)
    model.init()

    println("Train model....")
    model.setListeners(new ScoreIterationListener(1))
    for (i <- 0 until nEpochs) {
      model.fit(mnistTrain)
      println("*** Completed epoch {} ***", i)

      println("Evaluate model....")
      val eval = new Evaluation(outputNum)
      while(mnistTest.hasNext()){
        val ds = mnistTest.next()
        val output = model.output(ds.getFeatureMatrix(), false)
        eval.eval(ds.getLabels(), output)
      }
      println(eval.stats())
      mnistTest.reset()
    }
    println("****************Example finished********************")
  }
}
