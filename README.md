# (COLM 2024) Task Success is not Enough: Investigating the Use of Video-Language Models as Behavior Critics for Catching Undesirable Agent Behaviors

**Official codebase of the COLM 2024 paper:** *Task Success is not Enough: Investigating the Use of Video-Language Models as Behavior Critics for Catching Undesirable Agent Behaviors*

**Update: The paper has been accepted to the conference of language modeling (COLM 2024)**

**Please refer to the paper homepage for better visualization**: https://guansuns.github.io/pages/vlm-critic/

**The videos can be viewed at:** https://drive.google.com/drive/folders/1Fk3nJprLsLV5mkNAynwasLWN3hPZiZFK?usp=sharing

Note: The code has been refactored for better readability. If you encounter any problem, feel free to email lguan9@asu.edu.

![GPT-4V Critic Examples](assets/success_examples.png)

### Instructions
1. Create a conda environment and install dependency
```
conda create -n vlm-critic python=3.11
conda activate vlm-critic
pip install -r requirements.txt
```

2. OpenAI or Google API keys: Create a config file `api_config.json` under `src/config/` and put your API keys there. Please follow the format of `src/config/template_api_config.json`.


3. Download the dataset and meta-data Excel file:
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


4. Note: the primary objective of our benchmark dataset is to investigate the feasibility of using VLMs as behavioral critics. As a result, all testing samples are presented in the 'train' split, and there are no separate validation or test splits.


### Obtaining Critiques
We provide a minimal example that shows how to load our video dataset and obtain critiques from VLMs: `get_critiques.py`

---
All copyright by the authors.





