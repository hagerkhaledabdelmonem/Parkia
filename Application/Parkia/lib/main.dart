import 'package:curved_labeled_navigation_bar/curved_navigation_bar.dart';
import 'package:curved_labeled_navigation_bar/curved_navigation_bar_item.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/material.dart';
import 'package:parkia/Home/information.dart';
import 'package:parkia/Home/support.dart';
import 'package:image/image.dart' as img;
import 'Home/home_file.dart';
import 'constants.dart';

void main()
{
  runApp(MaterialApp(
    debugShowCheckedModeBanner: false,
    home: BottomNavBar(),
    // Splash_Screen(),
  ));
}
class BottomNavBar extends StatefulWidget {
  @override
  _BottomNavBarState createState() => _BottomNavBarState();
}

class _BottomNavBarState extends State<BottomNavBar> {
  int _page = 1;
  GlobalKey<CurvedNavigationBarState> _bottomNavigationKey = GlobalKey();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      bottomNavigationBar: CurvedNavigationBar(
        key: _bottomNavigationKey,
        index: 1,
        items: const [
          CurvedNavigationBarItem(
            child: Icon(Icons.info,color: whiteColor),
            label: 'Information',
          ),
          CurvedNavigationBarItem(
            child: Icon(Icons.home,color: whiteColor,),
            label: 'Home',
          ),
          CurvedNavigationBarItem(
            child: Icon(Icons.family_restroom,color: whiteColor),
            label: 'Support',
          ),
        ],
        color: lightBlue,
        buttonBackgroundColor: lightBlue,
        backgroundColor: whiteColor,
        animationCurve: Curves.easeInOut,
        animationDuration: const Duration(milliseconds: 600),
        onTap: (index) {
          setState(() {
            _page = index;
          });
        },
        letIndexChange: (index) => true,
      ),
      body: buildBody()
    );
  }

  buildBody() {
    switch (_page) {
      case 0:
        return Information();
      case 1:
        return HomePage();
      case 2:
        return Support();
      default:
      // Default to home page
        return HomePage();
    }
  }
}
