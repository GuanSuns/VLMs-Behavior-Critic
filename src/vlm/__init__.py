import traceback

import google.generativeai as genai
from wrapt_timeout_decorator import timeout
import tenacity


###################
## OpenAI Models ##
###################
@timeout(dec_timeout=30, use_signals=False)
def connect_openai(client, engine, messages, temperature, max_tokens,
                   top_p, frequency_penalty=None, presence_penalty=None):
    print(f'[INFO] Connecting to OpenAI engine {engine} ...')
    return client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )


def get_openai_response(end_when_error, max_retry,
                        client, engine, messages, temperature, max_tokens,
                        top_p, frequency_penalty=None, presence_penalty=None, verbose=False):
    vlm_output = None
    try:
        r = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(max_retry),
            wait=tenacity.wait_fixed(1.5),
            reraise=True
        )
        response = r.__call__(connect_openai, client=client,
                              engine=engine,
                              messages=messages,
                              temperature=temperature,
                              max_tokens=max_tokens,
                              top_p=top_p,
                              frequency_penalty=frequency_penalty,
                              presence_penalty=presence_penalty)
        vlm_output = response.choices[0].message.content
        if verbose:
            print(f'[INFO] Connection success - token usage: {response.usage}')
    except Exception as e:
        print(f'[ERROR] OpenAI model error: {e}')
        print(traceback.format_exc())
        if end_when_error:
            raise e
    return vlm_output


###################
## Google Models ##
###################
@timeout(dec_timeout=30, use_signals=False)
def connect_google(client, engine, messages, temperature, max_tokens, top_p,
                   frequency_penalty=None, presence_penalty=None):
    print(f'[INFO] Connecting to Google engine ...')
    response = client.generate_content(messages,
                                       generation_config=genai.types.GenerationConfig(
                                           candidate_count=1,
                                           top_p=top_p,
                                           max_output_tokens=max_tokens,
                                           temperature=temperature),
                                       stream=False)
    response.resolve()
    return response


def get_google_response(end_when_error, max_retry,
                        client, engine, messages, temperature, max_tokens,
                        top_p, frequency_penalty=None, presence_penalty=None, verbose=False):
    vlm_output = None
    try:
        r = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(max_retry),
            wait=tenacity.wait_fixed(1.5),
            reraise=True
        )
        response = r.__call__(connect_google, client=client,
                              engine=None,
                              messages=messages,
                              temperature=temperature,
                              max_tokens=max_tokens,
                              top_p=top_p)
        vlm_output = response.text
    except Exception as e:
        print(f'[ERROR] Google model error: {e}')
        print(traceback.format_exc())
        if end_when_error:
            raise e
    return vlm_output
