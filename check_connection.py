"""Kiểm tra Hub, Worker Token và contract v2.5; không in token."""
import os, sys
from urllib.parse import urlsplit
import requests


def check(url: str, token: str, timeout: int = 20):
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise ValueError('Hub URL phải có dạng https://ten-mien hoặc http://127.0.0.1:8000')
    if not token:
        raise ValueError('Thiếu Worker Token')
    response = requests.get(
        url.rstrip('/') + '/api/worker/config',
        headers={'Authorization': 'Bearer ' + token},
        timeout=timeout,
    )
    if response.status_code in (401, 403):
        raise RuntimeError('Token không đúng hoặc không có quyền Worker (HTTP %s)' % response.status_code)
    response.raise_for_status()
    data = response.json()
    for field in ('rules', 'skill', 'hash'):
        if field not in data:
            raise RuntimeError('Hub không đúng phiên bản: thiếu %s trong /api/worker/config' % field)
    return {'host': parsed.netloc, 'rules_hash': str(data['hash'])[:12], 'skill_available': bool(data['skill'])}


if __name__ == '__main__':
    import getpass
    address = os.environ.get('HUB_URL') or input('HUB_URL: ').strip()
    secret = os.environ.get('WORKER_TOKEN') or getpass.getpass('WORKER_TOKEN: ')
    try:
        print('Hub OK:', check(address, secret))
    except (ValueError, RuntimeError, requests.RequestException) as exc:
        print('Kết nối không thành công:', exc, file=sys.stderr)
        sys.exit(1)
