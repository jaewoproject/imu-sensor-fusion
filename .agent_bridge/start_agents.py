import os
import json
import logging
import urllib.request
import urllib.error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - AIOpsBridge - %(levelname)s - %(message)s')
logger = logging.getLogger("MultiAgentBridge")

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BRIDGE_DIR, "bridge_config.json")
RESPONSES_DIR = os.path.join(BRIDGE_DIR, "responses")

os.makedirs(RESPONSES_DIR, exist_ok=True)

def load_config():
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config file not found: {CONFIG_PATH}")
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def call_anthropic_api(api_key, model, prompt):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": model,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}]
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return res_data['content'][0]['text']
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode('utf-8')
        return f"API Error ({e.code}): {error_msg}"
    except Exception as e:
        return f"Request Exception: {str(e)}"

def call_openai_api(api_key, model, prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return res_data['choices'][0]['message']['content']
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode('utf-8')
        return f"API Error ({e.code}): {error_msg}"
    except Exception as e:
        return f"Request Exception: {str(e)}"

def run_agent_api(agent_name, agent_info, prompt_text):
    logger.info(f"Sending request to {agent_name} API...")
    
    api_key_env = agent_info.get("api_key_env", "")
    api_key = os.environ.get(api_key_env)
    
    if not api_key:
        return f"Error: API key environment variable '{api_key_env}' is not set. Please set it in your terminal."
        
    agent_type = agent_info.get("type")
    model = agent_info.get("model")
    
    if agent_type == "anthropic":
        return call_anthropic_api(api_key, model, prompt_text)
    elif agent_type == "openai":
        return call_openai_api(api_key, model, prompt_text)
    else:
        return f"Error: API agent type '{agent_type}' is not fully implemented yet."
