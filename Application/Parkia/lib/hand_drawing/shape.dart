import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'package:flutter/rendering.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/material.dart';
import 'package:parkia/speech/waveAudio.dart';
import 'package:screenshot/screenshot.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../constants.dart';
import 'dart:ui' as ui;
import 'package:flutter/services.dart';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

// Usage
class ShapeTest extends StatefulWidget {
  int index = 0;
  var imagesLink = ["assets/spiral.jpeg","assets/meander.jpeg","assets/circle.jpeg"];
  List<Uint8List> imageList = [];
  List<String> shapeName = ["Spiral","Meander","Circle"];
  ShapeTest({required this.index,required this.imageList});
  @override
  State<ShapeTest> createState() => _ShapeTestState();
}

class _ShapeTestState extends State<ShapeTest> with SingleTickerProviderStateMixin{
  bool _showCircle = true;
  ScreenshotController screenshotController = ScreenshotController();
  final GlobalKey _globalKey = GlobalKey();
  final _strokes = <Path>[];
  List<List<Object>>? output;
  Interpreter? interpreter;
  Uint8List? capturedImage;
  Timer? _hideTimer;
  @override
  // TODO: implement widget
  ShapeTest get widget => super.widget;
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: Duration(seconds: 3),
    )..addStatusListener((status) {
      if (status == AnimationStatus.completed) {
        setState(() {
          _showCircle = false;
        });
      }
    });

    // Start the animation when the screen is initialized
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
  @override
  void _startStroke(double x,double y){
    _strokes.add(Path()..moveTo(x, y));
  }
  void _moveStroke(double x,double y){
    setState(() {
      _strokes.last.lineTo(x, y);
    });
  }
  void _removeLastStroke() {
    setState(() {
      _strokes.clear();

    });
    (_globalKey.currentContext!.findRenderObject() as RenderRepaintBoundary).child!.markNeedsPaint();

  }
  Future<void> saveImageWithDrawing() async {
    // Find the RenderRepaintBoundary
    RenderRepaintBoundary boundary = _globalKey.currentContext!.findRenderObject() as RenderRepaintBoundary;
    // Convert the boundary to an image
    ui.Image image = await boundary.toImage(pixelRatio: 1.0);

    // Convert the image to bytes
    ByteData? byteData =
    await image.toByteData(format: ui.ImageByteFormat.png);
    // Extract the bytes as Uint8List
    capturedImage = byteData!.buffer.asUint8List();

    // print("image: $capturedImage");
  }
  @override
  Widget build(BuildContext context) {
    double width = MediaQuery.of(context).size.width;
    double w = width-11;
    double h = w*37/40 +20;
    String info_text = (widget.index == 2)?"Please draw a circle like the one shown a moment ago and repeat it more than once":"Please follow the trajectory line then press Next to continue";
    return Scaffold(
      backgroundColor: whiteColor,
      appBar: AppBar(
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
        title:Row(
          children: [
            const Spacer(flex: 2,),
            Text("${widget.shapeName[widget.index]} Test",
              style: const TextStyle(
                  color: darkestOne,
                  fontWeight: FontWeight.bold,
                  fontSize:20
              ),
            ),
            const Spacer(flex: 4,),

          ],
        ),

      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          const Spacer(flex: 1),
          InkWell(
            onTap: () {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20),side: const BorderSide(color: darkBlue,width: 4)),
                  backgroundColor: TextBoxColor,
                  content: Text(info_text,style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 15,color: darkBlue),),
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
          const Spacer(flex: 1,),
          GestureDetector(
            onPanDown: (details) => _startStroke(
                details.localPosition.dx,
                details.localPosition.dy),
            onPanUpdate: (details)=> _moveStroke(
                details.localPosition.dx,
                details.localPosition.dy),
            child: Container(
              width: w,
              height: h,
              padding: const EdgeInsets.all(3),
              decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(15),
                  border: Border.all(color: lightBlue,width: 4)
              ),
              child: RepaintBoundary(
                key: _globalKey,
                child: Stack(
                  children: [
                    Container(
                      width: w,
                      height: h,
                      decoration: BoxDecoration(
                        image: DecorationImage(
                          image:AssetImage(widget.imagesLink[widget.index]),
                          fit: BoxFit.fill,
                        ),
                      ),
                    ),
                    (_showCircle & (widget.index == 2))
                        ? Positioned(
                        left: w/2,
                        top: h/2,
                        child: AnimatedBuilder(
                          animation: _controller,
                          builder: (context, child) {
                            return CustomPaint(
                              painter: CirclePainter(_controller.value),
                            );
                          },
                        ))
                        : CustomPaint(
                          painter: DrawingPainter(_strokes),
                         ),

                  ],
                ),
              )
            ),
          ),
          const Spacer(flex: 2,),
          Row(
            children: [
              const Spacer(),
              ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 8,horizontal: 20),
                      backgroundColor: darkBlue,
                      side: const BorderSide(color: lightBlue,width: 3)
                  ),
                  onPressed: _removeLastStroke,
                  child: const Row(
                    children: [
                      Icon(Icons.edit,color: whiteColor,),
                      SizedBox(width: 4,),
                      Text("Redraw",style: TextStyle(color: whiteColor,fontSize: 18,fontWeight: FontWeight.bold),)],)
              ),
              const Spacer(),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 8,horizontal: 35),
                    backgroundColor: darkBlue,
                    side: const BorderSide(color: lightBlue,width: 3)
                ),
                child: const Row(
                  children: [
                    Icon(Icons.send_to_mobile,color: whiteColor,),
                    SizedBox(width: 3,),
                    Text("Next",style: TextStyle(color: whiteColor,fontSize: 18,fontWeight: FontWeight.bold))],),
                onPressed: () async{
                  await saveImageWithDrawing();
                  widget.imageList.add(capturedImage!);
                  if(widget.index == 2){
                    Navigator.pushReplacement(
                      context,
                      MaterialPageRoute(builder: (context) => AudioScreen(testData: widget.imageList,)),
                    );
                  }
                  else{
                    Navigator.pushReplacement(
                      context,
                      MaterialPageRoute(builder: (context) => ShapeTest(index: widget.index + 1,imageList: widget.imageList,)),
                    );
                  }
                },
              ),
              const Spacer(),
            ],
          ),
          const Spacer(flex: 2,),
        ],
      ),
    );
  }
}

class DrawingPainter extends CustomPainter
{
  final List<Path> strokes;
  DrawingPainter(this.strokes);
  @override
  void paint(Canvas canvas, Size size) {
    for(final stroke in strokes){
      final paint = Paint()
        ..style = PaintingStyle.stroke
        ..color = const Color(0xff4854cc)
        ..strokeWidth = 3;
      canvas.drawPath(stroke, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }

}
class CirclePainter extends CustomPainter {
  final double progress;

  CirclePainter(this.progress);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.black
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4.0;

    const radius = 150.0;
    final center = Offset(size.width / 2, size.height / 2);

    canvas.drawCircle(center, radius, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}