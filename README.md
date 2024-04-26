# "Task Success" is not Enough: Investigating the Use of Video-Language Models as Behavior Critics for Catching Undesirable Agent Behaviors

**Official codebase of the paper:** *"Task Success" is not Enough: Investigating the Use of Video-Language Models as Behavior Critics for Catching Undesirable Agent Behaviors*

**Please refer to the paper homepage for better visualization**: https://guansuns.github.io/pages/vlm-critic/

**The videos can be viewed at:** https://drive.google.com/drive/folders/1Fk3nJprLsLV5mkNAynwasLWN3hPZiZFK?usp=sharing

Note: The code has been refactored for better readability. If you encounter any problem, feel free to email lguan9@asu.edu.

![GPT-4V Critic Examples](assets/success_examples.png)

### Setup Environment
1. Create a conda environment and install dependency
```
conda create -n vlm-critic python=3.11
conda activate vlm-critic
pip install -r requirements.txt
```
2. Download the dataset and meta-data Excel file:
- Link: https://drive.google.com/drive/folders/125ZOcuL35TZWO2vdcw2OD9o569NPR6oK?usp=drive_link
- Download everything under the `videos` directory and place the `videos` folder under the root directory of the project. The project structure should look like this eventually:
```
.
|-- ...
|-- videos
|   |-- meta_data.xlsx
|   |-- manual
|       |-- train
|           |-- ...(*.mp4)
```
3. Note: the primary objective of our benchmark dataset is to investigate the feasibility of using VLMs as behavioral critics. As a result, all testing samples are presented in the 'train' split, and there are no separate validation or test splits.


### Obtaining Critiques
We provide a minimal example that shows how to load our video dataset and obtain critiques from VLMs: `get_critiques.py`

---
All copyright by the authors.





