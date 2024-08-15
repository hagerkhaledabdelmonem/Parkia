import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:parkia/constants.dart';

class Information extends StatelessWidget {
  const Information({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    double width = MediaQuery.of(context).size.width;
    double height = MediaQuery.of(context).size.height;
    return Scaffold(
      backgroundColor: darkBlue,
      body: Padding(padding: const EdgeInsets.fromLTRB(10, 70, 10, 0),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            color: whiteColor
        ),
        child: SingleChildScrollView( // Add SingleChildScrollView here
          child: Column(
            children: [
              const Text("- On each screen, click on the button (click for instruction) to find out what is required from you",
                style: TextStyle(
                    color: darkBlue,
                    fontSize: 15,
                    fontWeight: FontWeight.w500,
                    overflow: TextOverflow.clip
                ),
              ),
              ClipRRect(
                borderRadius: BorderRadius.circular(15),
                child: Image.asset("assets/click.jpg",height: 70,width: 250,),
              ),
              const Padding(padding: EdgeInsets.symmetric(vertical: 10),
                child: Divider(
                  thickness: 2,
                  indent: 10,
                  endIndent: 10,
                  color: darkBlue,
                ),),
              Row(
                children: [
                  const Expanded(
                    child: Text("- Once you start the test, you'll see three shapes, one after the other. In each of them there is a trajectory line that you must follow as closely as possible. With the exception of the last shape, which is the circle, a shape will appear and then disappear after three seconds, and you must draw something like it several times.",
                    style: TextStyle(
                        color: darkBlue,
                        fontSize: 15,
                        fontWeight: FontWeight.w500,
                        overflow: TextOverflow.clip
                     ),
                   ),
                  ),
                  const SizedBox(width: 10,),
                  Column(
                    children: [
                      Image.asset("assets/shape1.jpg",width: 100,height: 100,),
                      const SizedBox(height: 5,),
                      Image.asset("assets/shape2.jpg",width: 100,height: 100,),
                      const SizedBox(height: 5,),
                      Image.asset("assets/shape3.jpg",width: 100,height: 100,),
                    ],
                  )
                ],
              ),
              const Padding(padding: EdgeInsets.symmetric(vertical: 10),
              child: Divider(
                thickness: 2,
                indent: 10,
                endIndent: 10,
                color: darkBlue,
              ),),
              Row(
                children: [
                  const Expanded(
                    child: Text("- If you make an unintentional mistake and want to redraw, you can press this button:",
                      style: TextStyle(
                          color: darkBlue,
                          fontSize: 15,
                          fontWeight: FontWeight.w500,
                          overflow: TextOverflow.clip
                      ),
                    ),
                  ),
                  const SizedBox(width: 10,),
                  Image.asset("assets/redrow.jpg",width: 130,height: 100,),
                ],
              ),
              const Padding(padding: EdgeInsets.symmetric(vertical: 10),
                child: Divider(
                  thickness: 2,
                  indent: 10,
                  endIndent: 10,
                  color: darkBlue,
                ),),
              Row(
                children: [
                  const Expanded(
                    child: Text("- After that, there is an audio test. You must press the record button to start recording, then read the written text, then stop recording",
                      style: TextStyle(
                          color: darkBlue,
                          fontSize: 15,
                          fontWeight: FontWeight.w500,
                          overflow: TextOverflow.clip
                      ),
                    ),
                  ),
                  const SizedBox(width: 10,),
                  Column(
                    children: [
                      Image.asset("assets/start.jpg",width: 70,height: 70,),
                      const SizedBox(height: 7,),
                      Image.asset("assets/stop.jpg",width: 70,height: 70,),
                    ],
                  )
                ],
              ),
              const Padding(padding: EdgeInsets.symmetric(vertical: 10),
                child: Divider(
                  thickness: 2,
                  indent: 10,
                  endIndent: 10,
                  color: darkBlue,
                ),),
              const Text("- If you make an unintentional mistake and want to delete the audio, pause and continue, please use these buttons:",
                style: TextStyle(
                    color: darkBlue,
                    fontSize: 15,
                    fontWeight: FontWeight.w500,
                    overflow: TextOverflow.clip
                ),
              ),
             Row(
               mainAxisAlignment: MainAxisAlignment.center,
               children: [
                 Image.asset("assets/delete.jpg",width: 70,height: 70,),
                 const SizedBox(width: 7,),
                 Image.asset("assets/pause.jpg",width: 70,height: 70,),
                 const SizedBox(width: 7,),
                 Image.asset("assets/start.jpg",width: 70,height: 70,),
               ],
             )
            ],
          ),
        ),
      ),
      )
    );
  }
}
