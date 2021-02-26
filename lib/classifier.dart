import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class Classifier {
  Interpreter _interpreter;
  List<String> _labelList;

  Classifier() {
    _loadModel();
    _loadLabel();
  }

  void _loadModel() async {
    // _interpreter = await Interpreter.fromAsset('us_products_V1_1.tflite');
    _interpreter = await Interpreter.fromAsset('tf_lite_model_quant_rps.tflite');

    var inputShape = _interpreter.getInputTensor(0).shape;
    var outputShape = _interpreter.getOutputTensor(0).shape;
    print(inputShape);
    print(outputShape);
    print('Load Model - $inputShape / $outputShape');
  }

  void _loadLabel() async {
    final labelData =
        // await rootBundle.loadString('assets/us_products_V1_1.txt');
        await rootBundle.loadString('assets/rps.txt');
    final labelList = labelData.split('\n');
    _labelList = labelList;
    print(labelData);
    print(labelList);
    print('Load Label');
  }

  Future<img.Image> loadImage(String imagePath) async {
    var originData = File(imagePath).readAsBytesSync();
    var originImage = img.decodeImage(originData);
    print("Loading Image");
    return originImage;
  }

  Future<dynamic> runModel(img.Image loadImage) async {
    // var modelImage = img.copyResize(loadImage, width: 224, height: 224);
    var modelImage = img.copyResize(loadImage, width: 150, height: 150);
    // var modelInput = imageToByteListUint8(modelImage, 224);
    var modelInput = imageToByteListUint8(modelImage, 150);
    print("Run Model");

    //[1, 100000]
    // var outputsForPrediction = [List.generate(100000, (index) => 0.0)];
    var outputsForPrediction = [List.generate(3, (index) => 0.0)];
    print("Before $outputsForPrediction");
    _interpreter.run(modelInput.buffer, outputsForPrediction);
    // print("After ${outputsForPrediction[0][0]}");
    Map<int, double> map = outputsForPrediction[0].asMap();
    var sortedKeys = map.keys.toList()
      ..sort((k1, k2) => map[k2].compareTo(map[k1]));
    // double prediction = outputsForPrediction[0][0];
    // print("$prediction");
    List<dynamic> result = [];
    // if(prediction > 0)
    //   return "Uninfected";
    // else
    //   return "Infected";
    for (var i = 0; i < 3; i++) {
      result.add({
        'label': _labelList[sortedKeys[i]],
        'value': map[sortedKeys[i]],
      });
    }
    print("Result $result");
    return result;
  }

  Uint8List imageToByteListUint8(img.Image image, int inputSize) {
    var convertedBytes = Uint8List(1 * inputSize * inputSize * 3);
    var buffer = Uint8List.view(convertedBytes.buffer);

    int pixelIndex = 0;
    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = image.getPixel(i, j);
        buffer[pixelIndex++] = img.getRed(pixel);
        buffer[pixelIndex++] = img.getGreen(pixel);
        buffer[pixelIndex++] = img.getBlue(pixel);
      }
    }
    return convertedBytes.buffer.asUint8List();
  }

  // Uint8List imageToByteListFloat32(
  //     img.Image image, int inputSize, double mean, double std) {
  //   var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
  //   var buffer = Float32List.view(convertedBytes.buffer);
  //   int pixelIndex = 0;
  //   for (var i = 0; i < inputSize; i++) {
  //     for (var j = 0; j < inputSize; j++) {
  //       var pixel = image.getPixel(j, i);
  //       buffer[pixelIndex++] = (img.getRed(pixel) - mean) / std;
  //       buffer[pixelIndex++] = (img.getGreen(pixel) - mean) / std;
  //       buffer[pixelIndex++] = (img.getBlue(pixel) - mean) / std;
  //     }
  //   }
  //   return convertedBytes.buffer.asUint8List();
  // }

  // Float32List imageToByteListUint8(img.Image image, int inputSize) {
  //   var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
  //   var buffer = Float32List.view(convertedBytes.buffer);
  //
  //   int pixelIndex = 0;
  //   for (int i = 0; i < inputSize; i++) {
  //     for (int j = 0; j < inputSize; j++) {
  //       var pixel = image.getPixel(j, i);
  //       buffer[pixelIndex++] = img.getRed(pixel).toDouble();
  //       buffer[pixelIndex++] = img.getGreen(pixel).toDouble();
  //       buffer[pixelIndex++] = img.getBlue(pixel).toDouble();
  //     }
  //   }
  //   return convertedBytes.buffer.asFloat32List();
  // }
}
