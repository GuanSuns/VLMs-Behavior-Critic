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
        'episode_samples_subset': set(list(range(3)))
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
    episode_ids = behavior_dataset.episode_ids()

    # get critiques
    for episode_id in config.episode_samples_subset:
        episode_id = str(episode_id)
        if episode_id not in episode_ids:
            print(f'[WARNING] episode with id {episode_id} is not in the dataset, skipping ...')
            continue

        traj_data = behavior_dataset.get_trajectory(episode_id)
        rgb_frames, instruction = traj_data.img_obs, traj_data.instruction
        # get vlm outputs
        vlm_output = vlm_critic.get_response_video(rgb_frames, task_description=instruction)
        print('#' * 20)
        print(f'episode: {episode_id}, VLM critique:')
        print(vlm_output)


if __name__ == '__main__':
    main()
