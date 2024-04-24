import os


def get_openai_key():
    import json
    openai_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'api_config.json')
    with open(openai_config_file) as f:
        d = json.load(f)
    return d['openai']['api_key']


def get_google_key():
    import json
    openai_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'api_config.json')
    with open(openai_config_file) as f:
        d = json.load(f)
    return d['google']['api_key']


def set_openai_key():
    import openai
    api_key = get_openai_key()
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key
