# Master Thesis Repository

**Repository with the code for the thesis:** Data and computer center prediction of usage and cost: An interpretable machine learning approach.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

**Thesis objective:** Master thesis developed in collaboration with Novobanco. The objective is to use interpretable machine learning models to predict computational usage of the novobanco data center. In addition, we will develop a novel method using NLP techniques to explore the impact of human context on novobanco data center usage. 

### Built With

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]
 
## Organization of this repository

### Baseline Models
##
Code developed to run the baseline models used in this master thesis. 

- **Exponential Smoothing -** [Baseline_Models-ExponentialSmoothing.py]
- **Long short-term memory (LSTM) -** [Baseline_Models-LSTM.py]
- **Prophet -** [Baseline_Models-Prophet.py]
- **Seasonal Autoregressive Integrated Moving Average (SARIMA) -** [Baseline_Models-SARIMA.py]
- **Transformer -** [Baseline_Models-Transformer.py]

### Topic modelling + Sentiment analysis
##
Code developed for Topic modelling and Sentiment analysis

- **Get Tweets Program -** [Get_Tweets_Program.py] 
  - Program to collect tweets via Twitter API (In case of using this script, you need to insert your Twitter API token keys).

- **Pre-processing text analysis -** [Pre-processing Text.ipynb]
  - Program to pre-process text to Sentiment analysis and topic modelling (Cleaning Text, Tokenization, Reduce Text (Stopwords removal and Remove small words(<=2 characters)), Obtaining the stem words and pos tagging).

### Interpretable Models
##
Code developed to run the interpretable models.

- code1

### Documents and Reports
##
Master thesis documents and reports.


## Author

**Gonçalo Furtado Mateus**

* **Github -** [github/gfMateus99](https://github.com/gfMateus99)
* **Email -** goncalomateus99@gmail.com
* **LinkedIn -** https://www.linkedin.com/in/gonçalo-mateus/

## License
Copyright © Gonçalo Furtado Mateus, NOVA School of Science and Technology, NOVA University Lisbon, Novobanco.

The NOVA School of Science and Technology, the NOVA University Lisbon and Novobanco have the right, perpetual and without geographical boundaries, to file and publish this dissertation through printed copies reproduced on paper or on digital form, or by any other means known or that may be invented, and to disseminate through scientific repositories and admit its copying and distribution for non-commercial, educational or research purposes, as long as credit is given to the author and editor.

<p float="left" >

<div>
<img align="center" alt="alt_text" src="https://www.unl.pt/sites/default/files/nova_logo21_hp.png" data-canonical-src="https://www.unl.pt/sites/default/files/nova_logo21_hp.png" width="100" height="45" style="margin-top: 20px"/>  &nbsp;
<img align="center" alt="alt_text" src="https://www.meiosepublicidade.pt/wp-content/uploads/2021/02/logo-nova-school-of-science-and-technology.jpg" data-canonical-src="https://www.novobanco.pt/" width="120" height="80" />  &nbsp; 
<img align="center" src="https://cdn.myportfolio.com/52c9985fffa93e38c02cab522b6c8a04/2ab2b72c-4b71-47ae-95c6-89ac45f6bbb3_rw_1920.jpg?h=8c9e8cefaf642aa87853288a04eccc1d" data-canonical-src="https://cdn.myportfolio.com/52c9985fffa93e38c02cab522b6c8a04/2ab2b72c-4b71-47ae-95c6-89ac45f6bbb3_rw_1920.jpg?h=8c9e8cefaf642aa87853288a04eccc1d" width="190" height="60" />  &nbsp;
</div>
</p>

[Get_Tweets_Program.py]: <https://github.com/gfMateus99/Master_Thesis/blob/main/Sentiment%20analysis%20%2B%20Topic%20modelling/Get_Tweets_Program.py>
[Baseline_Models-ExponentialSmoothing.py]: <https://github.com/gfMateus99/Master_Thesis/blob/main/Baseline%20Models/Baseline_Models-ExponentialSmoothing.py>
[Baseline_Models-LSTM.py]: <https://github.com/gfMateus99/Master_Thesis/blob/main/Baseline%20Models/Baseline_Models-LSTM.py>
[Baseline_Models-Prophet.py]: <https://github.com/gfMateus99/Master_Thesis/blob/main/Baseline%20Models/Baseline_Models-Prophet.py>
[Baseline_Models-SARIMA.py]: <https://github.com/gfMateus99/Master_Thesis/blob/main/Baseline%20Models/Baseline_Models-SARIMA.py>
[Baseline_Models-Transformer.py]: <https://github.com/gfMateus99/Master_Thesis/blob/main/Baseline%20Models/Baseline_Models-Transformer.py>
[Pre-processing Text.ipynb]: <https://github.com/gfMateus99/Master_Thesis/blob/main/Sentiment%20analysis%20%2B%20Topic%20modelling/Pre-processing%20Text.ipynb>
   
