"""Loopback-only on-demand simulation server; no third-party web dependencies."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json,math,threading
from trial import run
from pack import pack
import mujoco as mj
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'results/config.json').read_text())
HTML=ROOT/'teardown-out/dist/tendonhand-physics.html'
LIMITS={'pull':(0,25),'thumb':(0,25),'mass':(.015,.120),'friction':(.2,2),'oppose':(-95,65)}
lock=threading.Lock()
class Handler(BaseHTTPRequestHandler):
    def reply(self,obj,code=200):
        b=json.dumps(obj,separators=(',',':'),allow_nan=False).encode()
        self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(b)
    def valid_host(self):return self.headers.get('Host') in ('127.0.0.1:18765','localhost:18765')
    def do_GET(self):
        if not self.valid_host():return self.reply({'error':'invalid host'},403)
        if self.path=='/api/status':return self.reply({'engine':'mujoco','version':mj.__version__})
        if self.path not in ('/','/index.html'):return self.reply({'error':'not found'},404)
        b=HTML.read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(b)
    def do_POST(self):
        if not self.valid_host() or self.headers.get('Origin') not in (None,'http://127.0.0.1:18765','http://localhost:18765'):return self.reply({'error':'invalid origin'},403)
        if self.path!='/api/run':return self.reply({'error':'not found'},404)
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.reply({'error':'JSON required'},415)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=2048:raise ValueError('invalid body length')
            data=json.loads(self.rfile.read(n))
            if not isinstance(data,dict) or set(data)!=set(LIMITS):raise ValueError('invalid fields')
            for k,v in data.items():
                if type(v) not in (int,float) or not math.isfinite(v) or not LIMITS[k][0]<=v<=LIMITS[k][1]:raise ValueError('invalid '+k)
        except (ValueError,TypeError):return self.reply({'error':'invalid simulation parameters'},400)
        if not lock.acquire(blocking=False):return self.reply({'error':'solver busy'},429)
        try:
            summary,frames=run(**dict(CFG,**data),duration=6,record=True)
            self.reply(pack({'summary':summary,'frames':frames}))
        except Exception as exc:self.reply({'error':type(exc).__name__},500)
        finally:lock.release()
if __name__=='__main__':
    print('TENDON/02 solver listening at http://127.0.0.1:18765',flush=True)
    ThreadingHTTPServer(('127.0.0.1',18765),Handler).serve_forever()
