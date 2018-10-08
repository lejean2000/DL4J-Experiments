package com.divvid.ai;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Properties;
import java.util.Random;

import javax.imageio.ImageIO;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AdaMax;

public class FashionMnistClassifier {
    private static final Logger log = LoggerFactory.getLogger(FashionMnistClassifier.class);
    /**
     * The location where the IDX files are.
     */
    private static String dataPath; 
    
    /**
     * Config file for your neural network.
     */
    private static String netConfigPath; 
    
    private static int seed = 64446;
    private static Random randNumGen;

    private static HashMap<String, Integer> config = new HashMap<>();
    
    private static void getPropValues() {
        var prop = new Properties();
     
        try (var input = new FileInputStream(netConfigPath);){
            // load the properties file
            prop.load(input);
     
            //task config - needs proper parsing with error checks
            config.put("height", Integer.parseInt(prop.getProperty("height")));
            config.put("width", Integer.parseInt(prop.getProperty("width")));
            config.put("channels", Integer.parseInt(prop.getProperty("channels")));
            config.put("outputNum", Integer.parseInt(prop.getProperty("outputNum")));// 10 categories of clothes
            config.put("batchSize", Integer.parseInt(prop.getProperty("batchSize")));
            config.put("nEpochs", Integer.parseInt(prop.getProperty("nEpochs")));
            
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
    
    public static MultiLayerNetwork getModel(){  
        var builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new AdaMax(0.001, 0.9, 0.999, 0.001))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(1)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(128)
                        .activation(Activation.LEAKYRELU)
                        .l2(0.01)
                        .build())
                .layer(new BatchNormalization.Builder()
                        .eps(0.001).build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .dropOut(0.75)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.LEAKYRELU)
                        .nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutionalFlat(28, 28, 1));
        
        var conf = builder.build();
        return new MultiLayerNetwork(conf);
    }

    @SuppressWarnings("unused")
    public static void getData(String dir, boolean train) {
        String inputImagePath  = dir;
        String inputLabelPath  = dir;
        String outputPath  = dir;

        if( train ){
            inputImagePath  += "train-images-idx3-ubyte";
            inputLabelPath  += "train-labels-idx1-ubyte";
            outputPath += "fmnist_png"+File.separator+"training"+File.separator;
            if (new File(outputPath).exists()) {
                log.info("Training data found!");
                return;
            }
        }else{
            inputImagePath += "t10k-images-idx3-ubyte";
            inputLabelPath += "t10k-labels-idx1-ubyte";
            outputPath += "fmnist_png"+File.separator+"testing"+File.separator;
            if (new File(outputPath).exists()) {
                log.info("Testing data found!");
                return;
            }
        }

        int[] hashMap = new int[10];

        try (
            var inImage = new FileInputStream(inputImagePath);
            var inLabel = new FileInputStream(inputLabelPath)
        )
        {
            int magicNumberImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfRows  = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfColumns = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());

            int magicNumberLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());
            int numberOfLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());

            var image = new BufferedImage(numberOfColumns, numberOfRows, BufferedImage.TYPE_INT_ARGB);
            int numberOfPixels = numberOfRows * numberOfColumns;
            int[] imgPixels = new int[numberOfPixels];

            String path;
            for(int i = 0; i < numberOfImages; i++) {

                if(i % 100 == 0) {log.info("Number of images extracted: {}",i);}

                for(int p = 0; p < numberOfPixels; p++) {
                    int gray = 255 - inImage.read();
                    imgPixels[p] = 0xFF000000 | (gray<<16) | (gray<<8) | gray;
                }

                image.setRGB(0, 0, numberOfColumns, numberOfRows, imgPixels, 0, numberOfColumns);

                int label = inLabel.read();

                hashMap[label]++;
                path = outputPath + label;
                new File(path).mkdirs();
                var outputfile = new File(path + File.separator + hashMap[label] + ".png");

                ImageIO.write(image, "png", outputfile);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Converts a folder structure to a RecordReaderDataSetIterator
     * @param imagesLocation
     * @return a vectorized DataSetIterator
     * @throws IOException e
     */
    private static DataSetIterator vectorize(String imagesLocation) throws IOException {
        var trainData   = new File(imagesLocation);
        var split       = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        var labelMaker  = new ParentPathLabelGenerator(); // parent path as the image label
        
        try(
            var rr = new ImageRecordReader(config.get("height"), config.get("width"), config.get("channels"), labelMaker);
        )
        {
            rr.initialize(split);
            return new RecordReaderDataSetIterator(rr, config.get("batchSize"), 1, config.get("outputNum"));
        }
    }

    public static void runModel() throws IOException {
        //Initialize the user interface backend
        var uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        var statsStorage = new InMemoryStatsStorage();

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        log.info("Vectorization...");

        // pixel values from 0-255 to 0-1 (min-max scaling)
        var scaler = new ImagePreProcessingScaler(0, 1);

        var trainIter = vectorize(dataPath + "fmnist_png"+File.separator+"training");
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);

        var testIter = vectorize(dataPath + "fmnist_png"+File.separator+"testing");
        testIter.setPreProcessor(scaler); // same normalization for better results

        log.info("Network configuration and training...");

        var net = getModel();

        net.init();
        net.setListeners(new StatsListener(statsStorage));

        log.debug("Total num of params: {}", net.numParams());

        // evaluation while training (the score should go down)
        for (int i = 0; i < config.get("nEpochs"); i++) {
            net.fit(trainIter);
            log.info("Completed epoch {}", i);
            var eval = net.evaluate(testIter);
            log.info(eval.stats());
            trainIter.reset();
            testIter.reset();
        }

        ModelSerializer.writeModel(net, new File(dataPath + "fmnist-model.zip"), true);

        log.info("Wrote final model");
    }

    public static void main(String[] args) throws IOException {
        dataPath = args[0];
        netConfigPath = args[1];
        randNumGen = new Random(seed);
        
        getPropValues();
        getData(dataPath, true);
        getData(dataPath, false);

        runModel();
    }
}