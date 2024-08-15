import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:parkia/results/result.dart';
import 'package:record/record.dart';
import 'package:scrollable_text_indicator/scrollable_text_indicator.dart';

import '../constants.dart';

class AudioScreen extends StatefulWidget {
  List<Uint8List> testData = [];
  AudioScreen({super.key, required this.testData});

  @override
  State<AudioScreen> createState() => _AudioScreenState();
}

class _AudioScreenState extends State<AudioScreen> {
  final audioRecorder = AudioRecorder();
  bool isRecording = false;
  bool isPaused = false;
  String? recordedFilePath;
  List<int>? audioData;
  double padding = 6.0;
  List<String> results = [];

  Future<void> startRecording() async {
    if (await audioRecorder.hasPermission()) {
      Directory tempDir = await Directory.systemTemp.createTemp();
      String path = '${tempDir.path}/recording.wav';

      RecordConfig config = const RecordConfig(
          encoder: AudioEncoder.wav,
          bitRate: 64000,
          sampleRate: 8000,
          numChannels: 1
      );
      await audioRecorder.start(
        config,
        path: path,
      );
      setState(() {
        isRecording = true;
      });
    } else {
      print("Permission to record audio is not granted.");
    }
  }
  Future<void> stopRecording() async {
    final path = await audioRecorder.stop();
    setState(() {
      isRecording = false;
      isPaused = false;
      recordedFilePath = path;
    });
    if (path != null) {
      print("Recording saved to: $path");
      await readWavFile(path);
      setState(() {
        if (results.isNotEmpty) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => ResultScreen(results: results)),
          );
        }
      });
    }
  }
  Future<void> pauseRecording() async {
    await audioRecorder.pause();
    setState(() {
      isRecording = false;
      isPaused = true;
    });
    print("Record paused....................");
  }
  Future<void> cancelRecording() async{
    audioRecorder.cancel();
    setState(() {
      isRecording = false;
      isPaused = false;
    });
    print("record canceled...........");
  }
  Future<void> continueRecording()async{
    audioRecorder.resume();
    setState(() {
      isRecording = true;
      isPaused = false;
    });
    print("continue recording................");
  }

  Future<void> readWavFile(String path) async {
    final file = File(path);
    final bytes = await file.readAsBytes();
    widget.testData.add(bytes);
    var request = http.MultipartRequest('POST', Uri.parse('http://192.168.1.24:5000/appTest'));
    request.files.add(http.MultipartFile.fromBytes(
      'image1', widget.testData[0],
      filename: 'screenshot1.png',
    ));
    request.files.add(http.MultipartFile.fromBytes(
      'image2', widget.testData[1],
      filename: 'screenshot2.png',
    ));
    request.files.add(http.MultipartFile.fromBytes(
      'image3', widget.testData[2],
      filename: 'screenshot3.png',
    ));
    request.files.add(http.MultipartFile.fromBytes(
      'audio', widget.testData[3],
      filename: 'audio.wav',
    ));
    // Send the request and get the response
    var response = await request.send();
    if (response.statusCode == 200) {
      var responseBody = await http.Response.fromStream(response);
      String body = responseBody.body;
      body = body.substring(4,body.length-3);
      results = body.split(",");

      // Handle the response from the server
      print(results);
    } else {
      print('Failed to send data: ${response.statusCode}');
    }
  }
  @override
  Widget build(BuildContext context) {
    double height = MediaQuery.of(context).size.height;
    return Scaffold(
      backgroundColor: whiteColor,
      appBar:AppBar(
        leading: Row(
          children: [
            IconButton(
                onPressed: (){
                  Navigator.pop(context);
                },
                icon: const Icon(
                  Icons.arrow_back_ios,
                  color: darkestOne,
                  weight: 30,
                  size: 25,
                )
            ),

          ],
        ),
        backgroundColor: Colors.transparent,
        title:const Row(
          children: [
            Spacer(flex: 2,),
            Text("Speech Test",
              style: TextStyle(
                  color: darkestOne,
                  fontWeight: FontWeight.bold,
                  fontSize:20
              ),
            ),
            Spacer(flex: 4,),
          ],
        ),
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const SizedBox(height: 20,),
          InkWell(
            onTap: () {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20),side: const BorderSide(color: darkBlue,width: 4)),
                  backgroundColor: TextBoxColor,
                  content: const Text("Please, start recording and read the text below then stop recording to see the result",style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15,color: darkBlue),),
                  duration: const Duration(seconds: 3), // Adjust the duration as needed
                  behavior: SnackBarBehavior.floating,
                  margin: const EdgeInsets.all(40),
                ),
              );
            },
            child: const Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  'Click for instructions',
                  style: TextStyle(color: midBlue, fontSize: 17, fontWeight: FontWeight.bold),),
                Icon(Icons.info_outline,color: midBlue,),
              ],
            ),
          ),
          const SizedBox(height: 10,),
          Flexible(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(5, 10, 5, 15),
              child: Container(
                height: 3 * height / 5,
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(15),
                    border: Border.all(color: lightBlue,width: 3)
                ),
                child: const ScrollableTextIndicator(
                  indicatorBarWidth: 1.5,
                  indicatorThumbColor: darkBlue,
                  indicatorBarColor: darkestOne,
                  indicatorThumbHeight: 40,
                  text: Text(
                    '''This is because there is less scattering of blue light as the atmospheric path length and consequently the degree of scattering of the incoming radiation is reduced. For the same reason, the sun appears to be whiter and less orange-coloured as the observer's altitude increases; this is because a greater proportion of the sunlight comes directly to the observer's eye. Figure 5.7 is a schematic representation of the path of electromagnetic energy in the visible spectrum as it travels from the sun to the Earth and back again towards a sensor mounted on an orbiting satellite. The paths of waves representing energy prone to scattering (that is, the shorter wavelengths) as it travels from sun to Earth are shown. To the sensor it appears that all the energy has been reflected from point P on the ground whereas, in fact, it has not, because some has been scattered within the atmosphere and has never reached the ground at all.\nNorth Wind and the Sun (Orthographic Version):\nThe North Wind and the Sun were disputing which was the stronger, when a traveler came along wrapped in a warm cloak. They agreed that the one who first succeeded in making the traveler take his cloak off should be considered stronger than the other. Then the North Wind blew as hard as he could, but the more he blew the more closely did the traveler fold his cloak around him; and at last the North Wind gave up the attempt. Then the Sun shone out warmly, and immediately the traveler took off his cloak. And so the North Wind was obliged to confess that the Sun was the stronger of the two.''',
                    style: TextStyle(
                      color: darkestOne,
                      fontWeight: FontWeight.w500,
                      fontSize: 18,
                    ),
                  ),
                ),
              ),
            ),
          ),
          Column(
            children: [
              Text(
                isRecording ? 'Recording...' : isPaused? 'press to continue': 'Press to Record',
                style: const TextStyle(fontSize: 24),
              ),
              isRecording?const SizedBox():const SizedBox(height: 9,),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  (isRecording || isPaused)?Column(
                    children: [
                      const SizedBox(height: 12,),
                      ElevatedButton(
                        onPressed: (){
                          cancelRecording();
                          setState(() {});
                        },
                        style: ElevatedButton.styleFrom(
                          shape: const CircleBorder(),
                          padding: const EdgeInsets.all(12),
                          backgroundColor: darkBlue,
                          side: const BorderSide(color: lightBlue,width: 3),
                        ),
                        child: const Icon(Icons.delete,size: 27,color: whiteColor,),
                      )
                    ],
                  ) :
                  const  SizedBox()
                  ,
                  const SizedBox(width: 5,),
                  ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      shape: const CircleBorder(),
                      padding: const EdgeInsets.all(12),
                      backgroundColor: darkBlue,
                      side: const BorderSide(color: lightBlue,width: 3),
                    ),
                    child: Icon(isRecording? Icons.pause : isPaused?Icons.mic_outlined:Icons.mic_none_outlined,size: 30,color: whiteColor,),
                    onPressed: ()async{
                      if(isRecording)
                      {
                        await pauseRecording();
                      }
                      else
                      {
                        if(isPaused)
                        {
                          await continueRecording();
                        }
                        else
                        {
                          await startRecording();
                        }
                      }
                      setState(() {});
                    },
                  ),
                  const SizedBox(width: 5,),
                  (isRecording || isPaused)?Column(
                    children: [
                      const SizedBox(height: 12,),
                      ElevatedButton(
                        onPressed: () async{
                          await stopRecording();
                        },
                        style: ElevatedButton.styleFrom(
                          shape: const CircleBorder(),
                          padding: const EdgeInsets.all(12),
                          backgroundColor: darkBlue,
                          side: const BorderSide(color: lightBlue,width: 3),
                        ),
                        child: const Icon(Icons.stop,size: 27,color: whiteColor),
                      )
                    ],
                  ) :
                  const SizedBox()
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }
}
