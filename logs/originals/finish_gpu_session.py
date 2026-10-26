"""Monitor the existing single T4 session; retrieve and stop after completion."""
import json,subprocess,time,sys
from pathlib import Path
CLI=r'C:\Users\drish\AppData\Local\Python\pythoncore-3.15-64\Scripts\colab.exe'
SESSION='patchkv-t4-experiment'
def call(args,code=None):
    return subprocess.run([CLI,*args],input=code,text=True,encoding='utf-8',errors='replace',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
status_code="""from pathlib import Path
import csv,json
base=Path('/content/patchkv_application')
f=base/'results.csv'
rows=list(csv.DictReader(f.open())) if f.exists() else []
exit_file=base/'experiment_exit.json'
print('GPU_STATUS '+json.dumps({'rows':len(rows),'exit_code':json.loads(exit_file.read_text())['exit_code'] if exit_file.exists() else None,'last':{k:rows[-1][k] for k in ['repetition','edit_number','condition','latency_s']} if rows else None}))
"""
while True:
    result=call(['exec','-s',SESSION,'--timeout','60'],status_code)
    lines=[line for line in result.stdout.splitlines() if line.startswith('GPU_STATUS ')]
    if not lines:
        print(result.stdout[-2500:],flush=True)
        raise RuntimeError('Could not monitor runtime; inspect before stopping')
    status=json.loads(lines[-1][11:]);print(json.dumps(status),flush=True)
    if status['exit_code'] is not None:break
    time.sleep(45)
archive_code="""from pathlib import Path
import csv,json,zipfile,hashlib
base=Path('/content/patchkv_application')
rows=list(csv.DictReader((base/'results.csv').open()))
expected={(str(r),c,str(e)) for r in range(1,4) for c in ['latest','naive_append'] for e in range(11)}
actual={(r['repetition'],r['condition'],r['edit_number']) for r in rows}
passed=len(rows)==66 and actual==expected and json.loads((base/'run_metadata.json').read_text())['status']=='complete' and json.loads((base/'smoke_test.json').read_text())['passed']
(base/'remote_verification.json').write_text(json.dumps({'passed':passed,'rows':len(rows),'complete_grid':actual==expected,'results_sha256':hashlib.sha256((base/'results.csv').read_bytes()).hexdigest()},indent=2))
with zipfile.ZipFile('/content/gpu_results.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in base.rglob('*'):
        if f.is_file() and '__pycache__' not in f.parts:z.write(f,str(f.relative_to(base)))
print('REMOTE_VERIFIED',passed,'ROWS',len(rows))
"""
result=call(['exec','-s',SESSION,'--timeout','60'],archive_code)
print(result.stdout,flush=True)
if result.returncode!=0:raise RuntimeError('Artifact archive failed; runtime remains available for recovery')
result=call(['download','-s',SESSION,'/content/gpu_results.zip','gpu_results.zip'])
print(result.stdout,flush=True)
if result.returncode!=0 or not Path('gpu_results.zip').exists():raise RuntimeError('Download failed; recover before stopping')
result=call(['stop','-s',SESSION]);print(result.stdout,flush=True)
Path('gpu_shutdown.txt').write_text(result.stdout,encoding='utf-8')
if result.returncode!=0:raise RuntimeError('Stop failed; inspect session status')
result=call(['sessions']);print(result.stdout,flush=True)
Path('colab_sessions_after_gpu.txt').write_text(result.stdout,encoding='utf-8')
Path('gpu_monitor_status.json').write_text(json.dumps(status,indent=2))
print('ARTIFACTS_DOWNLOADED_AND_SESSION_STOPPED',flush=True)
