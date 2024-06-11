import addict

from src.data.data_loader import VideoDatasetLoader
from src.vlm.vlm_critic import GPTsCritic
from src.vlm.vlm_critic import GoogleCritic


def main():
    """
    A minimal example showing how to obtain critiques from VLMs
    """
    config = addict.Dict({
        'vlm_model': 'gpt-4-vision-preview',    # 'gpt-4-vision-preview' or 'gemini-pro-vision'
        'dataset_name': 'manual',
        'dataset_split': 'train',   # only one split for now
        'test_cases_subset': set(list(range(3)))
    })

    # load VLM model
    if 'gpt' in config.vlm_model:
        vlm_critic = GPTsCritic(engine=config.vlm_model, temperature=0, verbose=True)
    elif 'gemini' in config.vlm_model:
        vlm_critic = GoogleCritic(engine=config.vlm_model, temperature=0, verbose=True)
    else:
        raise NotImplementedError

    # load the dataset
    behavior_dataset = VideoDatasetLoader(dataset_name=config.dataset_name, split=config.dataset_split)
    behavior_dataset.print_basic_info()

    # get critiques
    for test_case_idx in config.test_cases_subset:
        test_case_data = behavior_dataset.test_cases[test_case_idx]

        episode_id = test_case_data['episode_id']
        positive_samples = test_case_data['positive_examples']
        negative_samples = test_case_data['negative_examples']
        task_instruction = test_case_data['instruction']
        rgb_frames = behavior_dataset.get_traj_data(episode_id).img_obs

        print('#' * 20)
        print(f'test case: {test_case_idx}| episode: {episode_id}')
        print('frames shape: ', rgb_frames.shape)
        print(f'task instruction: {task_instruction}')

        # get vlm outputs
        vlm_output = vlm_critic.get_response_video(rgb_frames, task_description=task_instruction)
        print(f'VLM critique:')
        print(vlm_output)


if __name__ == '__main__':
    main()
