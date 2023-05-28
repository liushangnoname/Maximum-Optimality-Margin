# Maximum-Optimality-Margin
Experiment codes and data of Maximum Optimality Margin: A Unified Approach for Contextual Linear Programming and Inverse Linear Programming.

Description:
Main algorithm:
  DataGeneration.py: how to generate the synthetic data, including Fractional Knapsack and Shortest Path
  LearningMethod.py: contains our implementation of offline learning methods in our paper, including our MOM methods (named as MarginLearning)
  LinearProgramMethod.py: contains auxiliary algorithms of linear programs
  OnlineMethod.py: contains our implementation of online learning methods in our paper, including our MOM methods (named as MarginLearning)

Experiment setup:
  MarginLearningExperiment.ipynb: main setup of our experiments (Figure 1, 2, 3, 4, and 7)
  MarginLearningExperiment_Rebuttal.ipynb: main setup of the extra experiments (Figure 5 and 6)
  MakingFigures.ipynb and MarkingFigures_Rebuttal.ipynb: plot figures

Data:
  Attack_Power: folder that contains the experiment data on scale noise attack ("Loss_powerXX.txt"'s are the Figure 2 in the paper)
  Degree: folder that contains the experiment data on degree of data generation ("Loss_degreeX.txt"'s are the Figure 1 in the paper, "Loss_degree_X_Ker.txt"'s are the Figure 4 in the paper)
  Online: folder that contains the experiment data on online algorithms (Figure 7 in the paper)
  Rebuttal1_Degree: folder that contains the extra experiment on degree of data generation on Fractional Knapsack("Loss_degree_X.txt"'s are the Figure 5 in the paper)
  Rebutall2_Degree: folder that contains the extra experiment on degree of data generation on Shortest Path("Loss_degree_X.txt"'s are the Figure 6 in the paper)
  Sample_Complexity: folder that contains the experiment data on sample complexity ("Loss_NXXX.txt"'s are the Figure 3 in the paper)
