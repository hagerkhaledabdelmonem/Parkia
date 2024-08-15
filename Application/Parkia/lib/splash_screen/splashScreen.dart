import 'dart:async';
import 'package:flutter/material.dart';
import 'package:parkia/constants.dart';

import '../main.dart';

class Splash_Screen extends StatefulWidget {
  @override
  State<StatefulWidget> createState() => StartState();
}

class StartState extends State<Splash_Screen> {
  @override
  void initState() {
    super.initState();
    startTime();
  }

  startTime() async {
    var duration = Duration(seconds: 2);
    return new Timer(duration, route);
  }

  route() {
    Navigator.pushReplacement(
        context, MaterialPageRoute(builder: (context) => BottomNavBar()));
  }

  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: whiteColor,
      body: Center(
          child:Container(
            child: Image.asset("assets/splash.gif"),
          )),
    );
  }
}