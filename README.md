# Master Thesis Repository

**Repository with the code for the thesis:** Data and computer center prediction of usage and cost: An interpretable machine learning approach.

**Thesis objective:** Master thesis developed in collaboration with Novobanco. The objective is to use interpretable machine learning models to predict computational usage of the novobanco data center. In addition, we develop a novel method using NLP techniques to explore the impact of human context on novobanco data center usage. 

**Note:** This repository presents only the most important code scripts developed for the objective of this thesis. Some other scripts (for creating the plots, managing parts of data, and testing) are not shown in this script.

**Built With:** 

[![Python][Python.js]][Python-url] [![Jupyter][Jupyter.js]][Jupyter-url] 
 
## Organization of this repository

### Interpretable Models

- code1

### Baseline Models

- **Exponential Smoothing -** [Baseline_Models-ExponentialSmoothing.py]
- **Long short-term memory (LSTM) -** [Baseline_Models-LSTM.py]
- **Prophet -** [Baseline_Models-Prophet.py]
- **Seasonal Autoregressive Integrated Moving Average (SARIMA) -** [Baseline_Models-SARIMA.py]
- **Transformer -** [Baseline_Models-Transformer.py]

### Topic modelling + Sentiment analysis

- **Get Tweets Script -** [Get_Tweets_Program.py] 
  - Program to collect tweets via Twitter API (**Note:** In case of using this script, you need to insert your own Twitter API token keys).

- **Pre-processing text analysis -** [Pre-processing Text.ipynb]
  - Program to pre-process text to Sentiment analysis and topic modelling (Cleaning Text, Tokenization, Reduce Text (Stopwords removal and Remove small words(<=2 characters)), Obtaining the stem words and pos tagging).

- **Sentiment Analysis code -** [Sentiment Analysis.py]
  - Dictionary-based sentiment analysis using [SentiLex-PT] and [EMOTAIX.PT] dictionaries.

- **Topic Modelling model code -** [Topic Modelling.ipynb]
  - Script created to run topic modelling model (DMM with Gibbs Sampling).

### Documents and Reports
  - [Master thesis document].


## Author

**Gonçalo Furtado Mateus**

* **Github -** [github/gfMateus99](https://github.com/gfMateus99)
* **Email -** goncalomateus99@gmail.com
* **LinkedIn -** https://www.linkedin.com/in/gonçalo-mateus/

## License
Copyright © Gonçalo Furtado Mateus, NOVA School of Science and Technology, NOVA University Lisbon, Novobanco.

The NOVA School of Science and Technology, the NOVA University Lisbon and the Novobanco have the right, perpetual and without geographical boundaries, to file and publish this dissertation through printed copies reproduced on paper or on digital form, or by any other means known or that may be invented, and to disseminate through scientific repositories and admit its copying and distribution for non-commercial, educational or research purposes, as long as credit is given to the author and editor.

<p float="left" >

<div>
<img align="center" alt="alt_text" src="https://www.unl.pt/sites/default/files/nova_logo21_hp.png" data-canonical-src="https://www.unl.pt/sites/default/files/nova_logo21_hp.png" width="100" height="45" style="margin-top: 20px"/>  &nbsp;
<img align="center" alt="alt_text" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Novologofct2021.png/1280px-Novologofct2021.png" data-canonical-src="https://www.novobanco.pt/" width="120" height="80" />  &nbsp; 
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
[Topic Modelling.ipynb]: <https://github.com/gfMateus99/Master_Thesis/blob/main/Sentiment%20analysis%20%2B%20Topic%20modelling/Topic%20Modelling.ipynb>
[Sentiment Analysis.py]: <https://github.com/gfMateus99/Master_Thesis/blob/main/Sentiment%20analysis%20%2B%20Topic%20modelling/Sentiment%20Analysis.py>  
[SentiLex-PT]: <https://b2find.eudat.eu/dataset/b6bd16c2-a8ab-598f-be41-1e7aeecd60d3>  
[EMOTAIX.PT]: <https://portulanclarin.net/repository/browse/emotaixpt/c2c715c0b1b111ea803e02420a0004034aecafbdb25f4a9787e7a27c9da6bd6a/>  
[Master thesis document]: <https://github.com/gfMateus99/Master_Thesis/blob/main/Documents%20and%20Reports/Thesis_Final.pdf/>  

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Python.js]: https://img.shields.io/badge/Python-35495E?style=for-the-badge&logo=python&logoColor=blue
[Python-url]: https://www.python.org/

[Jupyter.js]: https://img.shields.io/badge/Jupyter_notebook-35495E?style=for-the-badge&logo=Jupyter&logoColor=orange
[Jupyter-url]: https://jupyter.org/

