package com.divvid.ai;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedHashMap;
import java.util.Map;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.model.VGG19;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A simple demonstration of how to use pretrained models
 * @author Ivan.Pavlov
 *
 */
public class VGGTest {
    private static final Logger log = LoggerFactory.getLogger(VGGTest.class);

    // This is for the Model Zoo support to load in the VGG16 model.
    public ComputationGraph loadVGG(int modeltype) throws IOException {
        log.info("Loading the model... Download is large 500+ MB, but will be cached afterwards.");

        ComputationGraph vgg = null;

        switch (modeltype) {
        case 16:
            vgg = (ComputationGraph) VGG16.builder().build().initPretrained(PretrainedType.IMAGENET);
            break;

        case 19:
            vgg = (ComputationGraph) VGG19.builder().build().initPretrained(PretrainedType.IMAGENET);
            break;
        }

        return vgg;

    }

    public Map<String, Double> classifyImageFileVGG(String filename, int modeltype, ComputationGraph vgg)
            throws IOException {
        var file = new File(filename);
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        var image = loader.asMatrix(file);

        if (modeltype == 16) {
            DataNormalization scaler = new VGG16ImagePreProcessor();
            scaler.transform(image);
        }
        INDArray[] output = vgg.output(false, image);

        return decodeVGGPredictions(output[0]);
    }

    // adapted from dl4j TrainedModels.VGG16 class.
    public Map<String, Double> decodeVGGPredictions(INDArray predictions) throws IOException {

        var recognizedObjects = new LinkedHashMap<String, Double>();
        String predictionDescription = "";
        var top5 = new int[5];
        var top5Prob = new float[5];
        var labels = new ImageNetLabels();

        // brute force collect top 5
        int i = 0;
        for (int batch = 0; batch < predictions.size(0); batch++) {
            if (predictions.size(0) > 1) {
                predictionDescription += String.valueOf(batch);
            }
            predictionDescription += " :";
            var currentBatch = predictions.getRow(batch).dup();
            while (i < 5) {
                top5[i] = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
                top5Prob[i] = currentBatch.getFloat(batch, top5[i]);
                recognizedObjects.put(labels.getLabel(top5[i]), (double) top5Prob[i]);
                currentBatch.putScalar(0, top5[i], 0);
                predictionDescription += "\n\t" + String.format("%3f", top5Prob[i] * 100) + "%, "
                        + labels.getLabel(top5[i]);
                i++;
            }
        }
        log.info(predictionDescription);
        return recognizedObjects;
    }

    public static void main(String[] args) throws IOException {
        // Non-default location of DL4J resources
        DL4JResources.setBaseDirectory(new File("D:\\tmp\\.deeplearning4j"));

        var t = new VGGTest();
        var v16 = t.loadVGG(16);
        var v19 = t.loadVGG(19);

        try (var br = new BufferedReader(new InputStreamReader(System.in))) {
            while (true) {
                System.out.print("Enter file path : ");
                String input = br.readLine();
                
                //continue if file does not exist
                try {
                    t.classifyImageFileVGG(input, 16, v16);
                    t.classifyImageFileVGG(input, 19, v19);
                } catch (IOException e) {
                    log.error(e.getMessage());
                    continue;
                }
            }
        }
    }
}
