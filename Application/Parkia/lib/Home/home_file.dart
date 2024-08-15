import 'package:flutter/material.dart';
import '../constants.dart';
import '../hand_drawing/shape.dart';

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  @override
  double padding = 6.0;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: whiteColor,
        appBar: AppBar(
          backgroundColor: Colors.transparent,
          title: const Center(child: Text("          Testing",
            style: TextStyle(
                color: darkestOne,
                fontWeight: FontWeight.bold,
                fontSize:20
            ),
          ),
          ),
          actions: const [Icon(Icons.menu,color: darkBlue,size: 35,),SizedBox(width: 15,)],
        ),
        body: Padding(
          padding: const EdgeInsets.fromLTRB(20, 30, 20, 30),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const Row(
                children: [
                  CircleAvatar(
                    backgroundImage: AssetImage("assets/user_avatar.jpg"),
                    radius: 35,
                  ),
                  SizedBox(
                    width: 15,
                  ),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("Hello",
                        style: TextStyle(
                          color: darkBlue,
                          fontSize: 17,
                          fontWeight: FontWeight.bold
                        ),
                      ),
                      Text("Have a nice day and don't forget to take\ncare of your health.",
                        style: TextStyle(
                          fontSize: 12,
                          color: darkestOne,
                          fontWeight: FontWeight.w500
                        ),
                      ),
                    ],
                  )
                ],
              ),
              const SizedBox(height: 40,),
              Container(
                height: 170,
                decoration: BoxDecoration(
                    color: darkBlue,
                    borderRadius: BorderRadius.circular(30),
                    boxShadow: const [
                      BoxShadow(
                        color: lightBlue,
                        blurRadius: 3,
                        offset: Offset(0,25),
                        spreadRadius: -10,
                      ),
                      BoxShadow(
                          color: midBlue,
                          blurRadius: 2,
                          offset: Offset(0,12),
                          spreadRadius: -5
                      ),
                    ]
                ),
                child:Row(
                  children: [
                    const Padding(
                      padding: EdgeInsets.only(left: 30 ,top: 30),
                      child:  Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text("Welcome!",
                            style: TextStyle(
                                color: whiteColor,
                                fontWeight: FontWeight.bold,
                                fontSize: 20
                            ),
                          ),
                          Text("It's time for \nParkinson's disease\nAwareness",
                            style: TextStyle(
                                color: whiteColor,
                                fontSize: 13,
                                fontWeight: FontWeight.w500
                            ),
                          )
                        ],
                      ),
                    ),
                    Spacer(),
                    Image.asset("assets/logo.png")
                  ],
                ),
              ),
              const Spacer(flex: 2,),
              const Text("Please, before start testing, read",
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: darkBlue,
                ),
              ),
              const Text("the instructions from ",
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: darkBlue,
                ),
              ),
              const Text("the information page below",
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: darkBlue,
                ),
              ),
              const Spacer(flex: 2,),
              GestureDetector(
                onTapDown: (_) => setState(() {
                  padding = 0.0;
                }),
                onTapUp: (_) {
                  setState(() {padding = 6.0;});
                  Navigator.push(
                      context,MaterialPageRoute(builder: (context) => ShapeTest(index: 0, imageList: []))
                  );
                },
                child: AnimatedContainer(
                  padding: EdgeInsets.only(bottom: padding),
                  decoration: BoxDecoration(
                      color: midBlue,
                      borderRadius: BorderRadius.circular(45),
                  ),
                  duration: const Duration(milliseconds: 100),
                  child: Container(
                    padding:  const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                        color: darkBlue,
                        border: Border.all(color: lightBlue,width: 4),
                      borderRadius: BorderRadius.circular(45)
                    ),
                    child: const Padding(
                      padding: EdgeInsets.symmetric( horizontal: 10),
                      child:  Text("Start Testing",style: TextStyle(color: whiteColor,fontWeight: FontWeight.bold,fontSize: 20),),
                    )
                  ),
                ),
              ),
              const Spacer(flex: 1,)
            ],
          ),
        )
    );
  }
}
