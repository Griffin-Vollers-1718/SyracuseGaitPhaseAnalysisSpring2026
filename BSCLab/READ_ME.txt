This Folder is for the expanded Neural Network Models that I plan on creating for the BSC Lab
This will take the work I have done in NNGaitAnalysisSpring2026 and make modifications
The modifications and changes will be dictated in Github as well as in this file

The current goal of my projects:
- Add robustness to the Binary Gait Phase detector by implementing either a reggressor algorithim or a timing check
- Adapt and train the Binary Estimator to include more subjects and speeds
- Implement new models to see if the preform better than just a straight DNN
    - After looking at the paper Ankle Exoskeleton Control via Data Driven Gait Estimation and Claude, I want to develop a 1D CNN - LSTM hybrid model to predict a binary classifier
      and regression of stance for my Machine Learning Final
    - What is different from that paper?
        - Limited to only angular velocity data/ angualar velocity plus accleration
        - First just accomplish stance/sing classifier and stance phase regression