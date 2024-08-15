import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:parkia/constants.dart';

class Support extends StatelessWidget {
  const Support({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      backgroundColor: whiteColor,
      body: Center(
        child: Text("Support Page",style: TextStyle(color: darkestOne,fontSize: 30,fontWeight: FontWeight.bold),),
      ),
    );
  }
}
