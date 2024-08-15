import 'package:auto_size_text/auto_size_text.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:parkia/constants.dart';

class Level extends StatefulWidget {
  List<String> results = [];
  int index =0;
  List<String> emojis = ["assets/level1.png","assets/level2.png","assets/level3.png"];
  List<String> status = ["You are not at risk for parkinson's disease",
                         "You may be in the first stage of Parkinson's disease, to be sure you should see a neurologist and do the necessary x-rays to find out ",
                         "You are now in the early stage of Parkinson's disease, and therefore you must visit a neurologist to begin treatment to control the disease and to avoid increasing its symptoms"];
  Level({required this.results});

  @override
  State<Level> createState() {
    return _LevelState();
  }
}

class _LevelState extends State<Level> {
  @override
  Widget build(BuildContext context) {
    var hand = widget.results[0].substring(1,widget.results[0].length-1);
    var audio = widget.results[1].substring(4,widget.results[1].length-1);
    if((hand == "Healthy") & (audio == "Healthy")) {
      widget.index = 0;
    }
    else if((hand == "Healthy") | (audio == "Healthy")) {
      widget.index = 1;
    }
    else {
      widget.index = 2;
    }
    return Padding(
        padding: const EdgeInsets.symmetric(vertical: 30),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Expanded(
          child: Text(widget.status[widget.index],
            style: const TextStyle(
                fontSize: 15,
                color: darkestOne,
                fontWeight: FontWeight.bold,
                overflow: TextOverflow.clip),
          ),
        ),
        const SizedBox(width: 10,),
        Image.asset(widget.emojis[widget.index],alignment: Alignment.center,width: 120,height: 120,)
      ],
    ),);
  }
}
