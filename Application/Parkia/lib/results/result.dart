import 'package:flutter/material.dart';
import 'package:parkia/results/level.dart';

import '../constants.dart';

class ResultScreen extends StatefulWidget {
  List<String> results = [];
  ResultScreen({super.key, required this.results});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  @override
  Widget build(BuildContext context) {
    double height = MediaQuery.of(context).size.height;
    return Scaffold(
      backgroundColor: lightBlue,
      appBar: AppBar(
        leading: Row(
          children: [
            IconButton(
                onPressed: () {
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
        title: const Row(
          children: [
            Spacer(flex: 2,),
            Text("Result",
              style: TextStyle(
                  color: darkestOne,
                  fontWeight: FontWeight.bold,
                  fontSize: 20
              ),
            ),
            Spacer(flex: 4,),
          ],
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Image.asset("assets/result_logo.png", fit: BoxFit.fill, height: 2 * height / 5),
            ),
            const SizedBox(height: 20,),
            const Text("  Result", style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold, color: darkestOne),),
            Expanded( // Use Expanded to make the container take up the remaining space
              child: Container(
                padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
                decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(12),
                    color: Colors.white
                ),
                child: SingleChildScrollView( // Add SingleChildScrollView here
                  child: Padding(
                    padding: const EdgeInsets.all(8.0), // Add padding to content inside the scrollable area
                    child: Column(
                      children: [
                        Level(results: widget.results),
                        const Divider(
                          thickness: 2,
                          indent: 10,
                          endIndent: 10,
                          color: darkBlue,
                        ),
                        const SizedBox( height: 15,),
                        const Text(
                          ''' Parkinson's disease is a progressive disorder that affects the nervous system and the parts of the body controlled by the nerves. Symptoms start slowly. The first symptom may be a barely noticeable tremor in just one hand. Tremors are common, but the disorder also may cause stiffness or slowing of movement.\nIn the early stages of Parkinson's disease, your face may show little or no expression. Your arms may not swing when you walk. Your speech may become soft or slurred. Parkinson's disease symptoms worsen as your condition progresses over time.\nAlthough Parkinson's disease can't be cured, medicines might significantly improve your symptoms. Occasionally, a health care professional may suggest surgery to regulate certain regions of your brain and improve your symptoms.''',
                          style: TextStyle(color: darkestOne, fontWeight: FontWeight.w500, fontSize: 13, overflow: TextOverflow.clip),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}