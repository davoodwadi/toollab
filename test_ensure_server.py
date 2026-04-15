import os
import time
import signal
import subprocess
import psutil
from openai import OpenAI, APIConnectionError

class Config:
    def __init__(self, model_name):
        self.model_name = model_name

class MockSession:
    def __init__(self, model_name):
        self.config = Config(model_name)
        self._client = OpenAI(
            base_url='http://127.0.0.1:8080/v1',
            api_key='none',
        )

    def _ensure_server(self):
        try:
            models = self._client.models.list()
            hosted_model = models.data[0].id
            if hosted_model == self.config.model_name:
                print("Server is already running with correct model.")
                return
            else:
                print(f"Model mismatch: hosted {hosted_model}, desired {self.config.model_name}. Killing server.")
                self._kill_llama_server()
                time.sleep(2)
        except APIConnectionError:
            pass

        print(f"Starting llama-server for {self.config.model_name}...")
        self._start_llama_server()

        for _ in range(120):
            try:
                models = self._client.models.list()
                if models.data and models.data[0].id == self.config.model_name:
                    print(f"Server is up and running {self.config.model_name}")
                    return
            except APIConnectionError:
                pass
            time.sleep(2)

        raise RuntimeError(f"Failed to start or connect to llama-server for {self.config.model_name}")

    def _kill_llama_server(self):
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'llama-server' in proc.info['name'] or (proc.info['cmdline'] and 'llama-server' in proc.info['cmdline'][0]):
                    os.kill(proc.info['pid'], signal.SIGKILL)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

    def _start_llama_server(self):
        llama_cache = os.environ.get('LLAMA_CACHE')
        if not llama_cache:
            for path in [os.path.expanduser('~/.lcpp_cache'), os.path.expanduser('~/data/.lcpp_cache'), '/home/dw/data/.lcpp_cache']:
                if os.path.exists(path):
                    llama_cache = path
                    break

        if not llama_cache or not os.path.exists(llama_cache):
            raise ValueError("Could not find LLAMA_CACHE directory")

        model_path = None
        for root, dirs, files in os.walk(llama_cache):
            if self.config.model_name in files:
                model_path = os.path.join(root, self.config.model_name)
                break

        if not model_path:
            raise ValueError(f"Model {self.config.model_name} not found in {llama_cache}")

        server_path = os.path.expanduser('~/llama.cpp/build/bin/llama-server')
        if not os.path.exists(server_path):
            raise FileNotFoundError(f"llama-server executable not found at {server_path}")

        parallel = "1" if "31B" in self.config.model_name else "4"
        cmd = [
            server_path,
            "-m", model_path,
            "-ngl", "999",
            "--parallel", parallel
        ]

        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    # Test with current model to see if it detects it
    # First, let's see what's running
    client = OpenAI(base_url='http://127.0.0.1:8080/v1', api_key='none')
    try:
        models = client.models.list()
        hosted_model = models.data[0].id
        print(f"Currently running: {hosted_model}")
    except Exception as e:
        hosted_model = "Qwen3.5-9B-UD-Q4_K_XL.gguf"
        print("Not running currently.")

    print(f"Testing with {hosted_model}")
    session = MockSession(hosted_model)
    session._ensure_server()
