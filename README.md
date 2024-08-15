# Early Detection of Parkinson's Disease using Hand Drawing and Speech
## Project Overview:

This project is part of my graduation thesis from the Faculty of Computer and Information Science, Scientific Computing Department, Ain Shams University.

## Objective
The early detection of Parkinson's disease by leveraging two key features: 
- <strong>Hand Drawing Analysis</strong> 
- <strong>Speech Analysis</strong>. 

The project has been implemented in a mobile application using <strong>Flutter</strong> for the front-end and <strong>Flask</strong> for the back-end.

### Key Features
<ul>
    <li><strong>Hand Drawing Analysis</strong>:
        <ul>
            <li>Users are prompted to draw three shapes: a spiral, a meander, and a circle. These shapes are then fused into one image with three channels.</li>
            <li>This combined image is fed into a <strong>Convolutional Neural Network (CNN)</strong> model to detect any indications of Parkinson's disease based on the irregularities in the drawing.</li>
        </ul>
    </li>
    <li><strong>Speech Analysis</strong>:
        <ul>
            <li>Users are required to read a text displayed on the screen, and their speech is recorded.</li>
            <li>The recorded audio is analyzed using a <strong>Random Forest model</strong> to identify potential signs of Parkinson's disease through speech pattern analysis.</li>
        </ul>
    </li>
</ul>

## Tech Stack:
- <strong>Mobile App:</strong> Flutter
- <strong>Back-End:</strong> Flask
- <strong>Machine Learning Models:</strong>
    - Convolutional Neural Network (CNN) for Hand Drawing Analysis
    - Random Forest for Speech Analysis

## Dataset:
<p>The datasets used for this project are publicly available and can be accessed as follows:</p>
<ul>
    <li><strong>Hand Drawing Analysis:</strong> We used the <a href="https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/" target="_blank">NewHandPD</a> dataset, which includes various hand-drawn shapes collected from Parkinson's patients and healthy individuals.</li>
    <li><strong>Speech Analysis:</strong> We utilized the dataset available at <a href="https://zenodo.org/records/2867216" target="_blank">Zenodo</a>, which contains audio recordings of individuals reading a set text aloud, used to analyze speech patterns indicative of Parkinson's disease.</li>
</ul>

<h2>App Screenshots</h2>
<p>Below are some screenshots of the mobile application:</p>
<img src="https://github.com/user-attachments/assets/66cb88f0-7360-487b-ac3e-deb92a3fec8c" alt="App Screenshot" width="300">

<h2>Usage</h2>
<ol>
    <li><strong>Hand Drawing Analysis:</strong>
        <ul>
            <li>Open the app and navigate to the drawing screen.</li>
            <li>Draw the required shapes: spiral, meander, and circle.</li>
            <li>Submit the drawing for analysis.</li>
        </ul>
    </li>
    <li><strong>Speech Analysis:</strong>
        <ul>
            <li>Read the text displayed on the screen aloud.</li>
            <li>The app will automatically record and analyze your speech.</li>
        </ul>
    </li>
    <li><strong>Results:</strong>
        <ul>
            <li>The app will provide a probability score indicating the likelihood of Parkinson's disease.</li>
        </ul>
    </li>
</ol>

<h2>Acknowledgments</h2>
<ul>
    <li><strong>Advisors:</strong> Manal Tantawi and Manar Sultan</li>
    <li><strong>University:</strong> Ain Shams University, Faculty of Computer and Information Science, Scientific Computing Department</li>
</ul>
