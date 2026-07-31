import urllib.request, re

resp = urllib.request.urlopen('http://localhost:8000/')
h = resp.read().decode('utf-8')

# Find all script blocks
matches = list(re.finditer(r'<script[^>]*>(.*?)</script>', h, re.DOTALL))
print(f'Script blocks: {len(matches)}')
for i, m in enumerate(matches):
    s = m.group(0)
    if 'dash-data' in s:
        print(f'  Script {i}: embedded data ({len(s)} chars)')
    else:
        print(f'  Script {i}: {len(s)} chars')
        if '\\\\' in s:
            print('    WARNING: double backslashes found')
        # Check for toggleRow issue
        if 'toggleRow' in s:
            idx = s.find('toggleRow')
            print(f'    toggleRow: {s[idx:idx+120]}')

# Check page render status
print()
print(f'JS STARTED in page: {"JS STARTED" in h}')
print(f'PAGE LOADED in page: {"PAGE LOADED" in h}')
print(f'RENDER OK in page: {"RENDER OK" in h}')

# Check for key JS initialization
if "document.getElementById('loadBanner').textContent = 'JS STARTED'" in h:
    print('First JS line is: JS STARTED update')
else:
    print('First JS line NOT found - might be cached old version')
