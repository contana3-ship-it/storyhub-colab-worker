"""Launcher độc lập. Chỉ yêu cầu HUB_URL và WORKER_TOKEN."""
import argparse
import getpass
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description='Story Dataset Hub Colab Worker')
    parser.add_argument('--hub-url', default=os.getenv('HUB_URL', ''))
    parser.add_argument('--model', default=os.getenv('MODEL_ID', 'Qwen/Qwen2.5-3B-Instruct'))
    parser.add_argument('--engine', default=os.getenv('ENGINE', 'transformers'), choices=['transformers', 'ollama', 'mock'])
    parser.add_argument('--check-only', action='store_true')
    opts = parser.parse_args()
    url = opts.hub_url or input('Hub URL HTTPS: ').strip()
    token = os.getenv('WORKER_TOKEN') or getpass.getpass('Worker Token: ')
    from check_connection import check
    result = check(url, token)
    print('Đã xác thực Hub:', result['host'], 'Rules:', result['rules_hash'])
    if opts.check_only:
        return 0
    env = dict(os.environ, HUB_URL=url, WORKER_TOKEN=token, MODEL_ID=opts.model, ENGINE=opts.engine)
    return subprocess.call([sys.executable, os.path.join(os.path.dirname(__file__), 'worker.py')], env=env)


if __name__ == '__main__':
    sys.exit(main())
