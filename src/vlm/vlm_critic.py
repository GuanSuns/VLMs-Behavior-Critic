import addict
import numpy as np
import openai
import google.generativeai as genai
from langchain.prompts import PromptTemplate

from src.utils.video_utils import preprocess_traj_img
from src.config import get_openai_key, get_google_key
from src.vlm import get_openai_response, get_google_response


class VLMsCritic:
    def __init__(self, engine, stop=None,
                 max_tokens=512, temperature: float = 0, top_p=1,
                 frequency_penalty=0.0, presence_penalty=0.0,
                 video_target_shape=None, fit_black_background=None,
                 frame_encode_format=None, max_frame=30, verbose=False):
        self.model_conn_func = None  # to be init by child class
        self.model_client = None  # to be init by child class
        self.get_response_func = None  # to be init by child class
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.freq_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.video_target_shape = video_target_shape
        self.fit_black_background = fit_black_background
        self.frame_encode_format = frame_encode_format
        self.max_frame = max_frame
        self.verbose = verbose

        self.critic_prompt_template = (
            '**Task:**\n'
            'You will be presented with a short video clip (a sequence of images) of a '
            'robot or human performing certain household tasks. Your task is to identify '
            'any undesired, potentially harmful, or risky behaviors in the video.\n'
            '\n**Example Responses (you should follow the same output format):**\n'
            '{examples}\n'
            '**Note**\n'
            '{note}\n'
            '**Now, please look at the following sequence of images carefully.**{extra}\n'
        )
        self.critic_prompt_template = PromptTemplate.from_template(self.critic_prompt_template)

        self.critic_prompt_default_examples = (
            'Example 1.\n'
            '## Has undesired behavior(s): Yes\n'
            '## What are the behavior(s):\n'
            '(a) The robot\'s arm holding the cup moved too fast, causing water to splash out.\n'
            '(b) The robot\'s gripper accidentally dragged the pot while trying to grasp the carrot inside, causing '
            'the pot to shift.\n'
            'Example 2.\n'
            '## Has undesired behavior(s): No\n'
            '## What are the behavior(s): N/A\n'
        )
        self.critic_prompt_default_note = (
            '1. The images are key frames extracted from a video and are presented in the same order as the '
            'original footage.\n'
            '2. In principle, humans or robots should not cause any unnecessary collisions between objects '
            'unless they are required in certain tasks.\n'
            '3. There may be zero, one, two or more undesired behaviors in a video.\n'
        )

        grounding_feedback_prompt_template = (
            'A more specialized perception model has been used to validate the events mentioned in the listed '
            'undesired behaviors. You can assume the information provided below is accurate. Please refine your '
            'previous output (i.e., the list of undesired behaviors) based on the provided information. '
            'Please remember to follow the output format as instructed earlier.\n'
            '{feedback_list}'
        )
        self.grounding_feedback_prompt_template = PromptTemplate.from_template(grounding_feedback_prompt_template)

    def get_response(self, messages, end_when_error=False, max_retry=5):
        return self.get_response_func(end_when_error=end_when_error, max_retry=max_retry,
                                      client=self.model_client, engine=self.engine, messages=messages,
                                      temperature=self.temperature, max_tokens=self.max_tokens,
                                      top_p=self.top_p, frequency_penalty=self.freq_penalty,
                                      presence_penalty=self.presence_penalty)

    def preprocess_frames(self, frames):
        key_frames, num_frames = None, len(frames)
        if num_frames > self.max_frame:
            # noinspection PyTypeChecker
            key_frames = np.linspace(start=0, stop=num_frames - 1, num=self.max_frame,
                                     endpoint=True, dtype=np.int16).tolist()
        return preprocess_traj_img(frames, target_shape=self.video_target_shape,
                                   fit_black_background=self.fit_black_background,
                                   key_frames=key_frames, encode_format=self.frame_encode_format)

    def get_text_prompt(self, task_description=None):
        extra_note = ''
        if task_description is not None:
            extra_note = f'\nThe robot is performing the task "{task_description.lower().strip(" .")}".'
        critic_text_prompt = self.critic_prompt_template.format(
            examples=self.critic_prompt_default_examples,
            note=self.critic_prompt_default_note, extra=extra_note)
        return critic_text_prompt

    def get_video_prompt_messages(self, frames, task_description=None):
        raise NotImplementedError

    def get_response_video(self, frames, task_description=None, end_when_error=False, max_retry=5):
        messages = self.get_video_prompt_messages(frames, task_description)
        return self.get_response(messages, end_when_error=end_when_error, max_retry=max_retry)

    def _append_message(self, messages, new_message):
        """
        This function should append message of two *roles* (for now):
            - assistant (a past response)
            - grounding_feedback
        A new_message should have the three keys below:
            - role (assistant or grounding_feedback)
            - type (only support 'text' now)
            - content
        """
        raise NotImplementedError

    def get_response_video_with_feedback(self, frames, history_and_feedback,
                                         task_description=None, end_when_error=False, max_retry=5):
        messages = self.get_video_prompt_messages(frames, task_description)
        for new_message in history_and_feedback:
            self._append_message(messages, new_message)
        return self.get_response(messages, end_when_error=end_when_error, max_retry=max_retry)

    @staticmethod
    def parse_vlm_output(output_str):
        if output_str is None:
            print('[WARNING] Got None as input at parse_vlm_output')
            return None

        parsed_output = addict.Dict({'has_undesired': None,
                                     'undesired_behaviors': list(),
                                     'raw_output': str(output_str)})
        output_str = output_str.strip()
        # parse 'has_undesired' prediction
        undesired_keywords = '## Has undesired behavior(s):'
        if undesired_keywords in output_str:
            has_undesired = output_str.split(undesired_keywords)[1].split('\n')[0].strip()
            if has_undesired.lower() in ['yes', 'no']:
                parsed_output['has_undesired'] = has_undesired.lower()
        # parse the list of undesired behaviors
        behavior_list_keywords = '## What are the behavior(s):'
        if parsed_output['has_undesired'] == 'yes' and behavior_list_keywords in output_str:
            behavior_list = output_str.split(behavior_list_keywords)[1].strip().split('\n')
            parsed_behaviors = list()
            for i, behavior_candidate in enumerate(behavior_list):
                behavior_candidate = behavior_candidate.strip()
                char_idx = f'({chr(ord("a") + i)})'
                if len(behavior_candidate) > len(char_idx) and char_idx == behavior_candidate[:len(char_idx)]:
                    parsed_behaviors.append(behavior_candidate[len(char_idx):].strip())
                elif len(behavior_candidate) > 0:
                    print(f'[INFO] Missing alphabet idx at the beginning of critique entry: {behavior_candidate}')
                    parsed_behaviors.append(behavior_candidate)
                else:
                    break
            parsed_output['undesired_behaviors'] = parsed_behaviors
        return parsed_output

    @staticmethod
    def format_critique_output(behavior_list):
        """
        This function recovers the VLM output given a list of undesired behaviors (i.e., a list of parsed output)
        """
        template = ('## Has undesired behavior(s): {has_undesired}\n'
                    '## What are the behavior(s):\n{behavior_list}')
        template = PromptTemplate.from_template(template)

        if behavior_list is None or len(behavior_list) == 0:
            has_undesired, behavior_list = 'No', 'N/A'
        else:
            behavior_list_str = ''
            for i, behavior in enumerate(behavior_list):
                idx, behavior = chr(ord('a') + i), behavior.strip('. \n')
                if i > 0:
                    behavior_list_str += '\n'
                behavior_list_str += f'({idx}) {behavior}.'
            has_undesired, behavior_list = 'Yes', behavior_list_str
        return template.format(has_undesired=has_undesired, behavior_list=behavior_list)


###################
## OpenAI Models ##
###################
class GPTsCritic(VLMsCritic):
    def __init__(self, engine, stop=None, max_tokens=512, temperature=0.2, top_p=1, frequency_penalty=0.0,
                 presence_penalty=0.0, video_target_shape=None, fit_black_background=None,
                 frame_encode_format='base64', max_frame=30, verbose=False, img_detail='low'):
        super().__init__(engine, stop, max_tokens, temperature, top_p, frequency_penalty, presence_penalty,
                         video_target_shape, fit_black_background, frame_encode_format, max_frame, verbose)
        self.img_detail = img_detail
        self.model_client = openai.OpenAI(api_key=get_openai_key())
        self.get_response_func = get_openai_response

    def _append_message(self, messages, new_message):
        """
        This function should append two types of message (for now):
            - assistant (a past response)
            - grounding_feedback
        A new_message should have the three keys below:
            - role (assistant or grounding_feedback)
            - type (only support 'text' now)
            - content
        """
        assert new_message['type'] == 'text', f'unsupported message type {new_message["type"]}'

        if new_message['role'] == 'assistant':
            messages.append({
                "role": "assistant",  # assistant message
                "content": [
                    {
                        "type": new_message['type'],
                        "text": new_message['content']
                    }
                ],
            })
        elif new_message['role'] == 'grounding_feedback':
            messages.append({
                "role": "user",  # user message
                "content": [
                    {
                        "type": new_message['type'],
                        "text": self.grounding_feedback_prompt_template.format(feedback_list=new_message['content'])
                    }
                ],
            })
        else:
            raise NotImplementedError

    def get_video_prompt_messages(self, frames, task_description=None):
        frames = self.preprocess_frames(frames)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.get_text_prompt(task_description),
                    }
                ],
            },
        ]
        for frame in frames:
            frame_message = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}",
                    "detail": self.img_detail
                }
            }
            messages[0]['content'].append(frame_message)
        return messages


###################
## Google Models ##
###################
class GoogleCritic(VLMsCritic):
    def __init__(self, engine='gemini-pro-vision', stop=None, max_tokens=512, temperature=0, top_p=1,
                 frequency_penalty=0.0, presence_penalty=0.0, video_target_shape=None, fit_black_background=None,
                 frame_encode_format='pil_jpg', max_frame=16, verbose=False, img_detail='low'):
        super().__init__(engine, stop, max_tokens, temperature, top_p, frequency_penalty, presence_penalty,
                         video_target_shape, fit_black_background, frame_encode_format, max_frame, verbose)
        genai.configure(api_key=get_google_key())
        self.model_client = genai.GenerativeModel(engine)
        self.get_response_func = get_google_response

        # Gemini doesn't support chat mode at this moment, so we override the feedback prompt
        self.grounding_feedback_prompt_template = ('\n\nA more specialized perception model has been used to provide '
                                                   'auxiliary information about the video. You can assume the '
                                                   'information provided below is accurate. You can leverage this '
                                                   'information for better capturing undesirable behaviors '
                                                   'present in the video.\n{feedback_list}')
        self.grounding_feedback_prompt_template = PromptTemplate.from_template(self.grounding_feedback_prompt_template)

    def _append_message(self, messages, new_message):
        """
        Gemini is not optimized for chat mode, so we only append
            grounding feedback as auxiliary information and discard past responses
        """
        assert new_message['type'] == 'text', f'unsupported message type {new_message["type"]}'

        if new_message['role'] == 'assistant':
            pass
        elif new_message['role'] == 'grounding_feedback':
            messages.append(self.grounding_feedback_prompt_template.format(feedback_list=new_message['content']))
        else:
            raise NotImplementedError

    def get_video_prompt_messages(self, frames, task_description=None):
        frames = self.preprocess_frames(frames)
        messages = [self.get_text_prompt(task_description), *frames]
        return messages
