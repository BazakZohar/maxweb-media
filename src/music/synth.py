import numpy as np, scipy.signal as ss, sys
from scipy.io import wavfile
SR=44100; DUR=float(sys.argv[1]) if len(sys.argv)>1 else 14.0
STYLE=sys.argv[3] if len(sys.argv)>3 else 'a'
# a = modern electronic (original), b = chill lo-fi, c = energetic house
CFG={'a':dict(bpm=118,prog=[[57,60,64],[53,57,60],[48,52,55],[55,59,62]],bass=[45,41,48,43],padcut=1800,kick=0.9,clap=0.45,arp=0.09,swing=0.0,openhat=True,offbass=True,seed=7),
     'b':dict(bpm=92,prog=[[50,53,57,60],[55,58,62,65],[48,52,55,59],[53,57,60,64]],bass=[38,43,36,41],padcut=1100,kick=0.75,clap=0.30,arp=0.06,swing=0.12,openhat=False,offbass=False,seed=11),
     'c':dict(bpm=124,prog=[[52,55,59],[48,52,55],[55,59,62],[50,54,57]],bass=[40,36,43,38],padcut=2600,kick=1.0,clap=0.5,arp=0.10,swing=0.0,openhat=True,offbass=True,seed=23)}[STYLE]
BPM=CFG['bpm']; beat=60/BPM
N=int(SR*DUR); L=np.zeros(N); R=np.zeros(N)
rng=np.random.default_rng(CFG['seed'])
def add(sig,t,gain=1.0,pan=0.0):
    i=int(t*SR); 
    if i>=N: return
    s=sig[:N-i]*gain; L[i:i+len(s)]+=s*(1-max(pan,0)); R[i:i+len(s)]+=s*(1+min(pan,0))
def env(n,a,d):
    t=np.arange(n)/SR; e=np.minimum(t/max(a,1e-4),1)*np.exp(-t/d); return e
def kick():
    n=int(.45*SR); t=np.arange(n)/SR; f=48+110*np.exp(-t*28); ph=2*np.pi*np.cumsum(f)/SR
    s=np.sin(ph)*np.exp(-t*7.5); s+=0.25*np.sin(ph*2)*np.exp(-t*30); return np.tanh(s*1.6)
def hat(open_=False):
    n=int((.18 if open_ else .05)*SR); s=rng.standard_normal(n); b,a=ss.butter(4,7500/(SR/2),'high'); s=ss.lfilter(b,a,s)
    return s*env(n,.001,.06 if open_ else .015)
def clap():
    n=int(.25*SR); s=rng.standard_normal(n); b,a=ss.butter(2,[1200/(SR/2),5000/(SR/2)],'band'); s=ss.lfilter(b,a,s)
    e=np.zeros(n)
    for o in (0,.012,.024): i=int(o*SR); e[i:]+=np.exp(-np.arange(n-i)/SR/.04)
    return s*e*0.6
def saw(f,n,det=0):
    t=np.arange(n)/SR; out=np.zeros(n)
    for d in ([0] if not det else [-det,0,det]):
        ff=f*(1+d); out+=2*((t*ff)%1)-1
    return out/ (1 if not det else 3)
def lp(s,fc,o=2): b,a=ss.butter(o,min(fc,SR/2-100)/(SR/2)); return ss.lfilter(b,a,s)
def note(m): return 440*2**((m-69)/12)
# chord progression Am - F - C - G (modern minor pop), 2 beats? use 1 bar each (4 beats)
prog=CFG['prog']; bass=CFG['bass']
bar=4*beat; nb=int(np.ceil(DUR/bar))
# pads with sidechain
pad=np.zeros(N)
for b in range(nb):
    ch=prog[b%4]; n=int(bar*SR); s=sum(saw(note(m),n,0.006) for m in ch)/len(ch)
    s=lp(s,CFG['padcut'])*env(n,.25,6); i=int(b*bar*SR); seg=s[:max(0,N-i)]; pad[i:i+len(seg)]+=seg
duck=np.ones(N)
for k in range(int(DUR/beat)+1):
    i=int(k*beat*SR); n=int(beat*SR); t=np.arange(n)/SR; d=1-0.75*np.exp(-t/0.09); seg=d[:max(0,N-i)]; duck[i:i+len(seg)]*=seg
pad*=duck
L+=pad*0.22; R+=pad*0.22
# bass: offbeat 8ths
for b in range(nb):
    for e in range(8):
        t0=b*bar+e*beat/2
        if CFG['offbass'] and e%2==0: continue
        if not CFG['offbass'] and e not in (0,3,6): continue
        n=int(beat/2*SR*0.9); s=lp(saw(note(bass[b%4]),n),500)*env(n,.005,.12)
        add(np.tanh(s*1.5),t0,0.32)
# pluck arpeggio (karplus-strong) 16ths
def pluck(f,dur=.35):
    n=int(dur*SR); p=int(SR/f); buf=rng.uniform(-1,1,p); out=np.zeros(n)
    for i in range(n): out[i]=buf[i%p]; buf[i%p]=0.5*(buf[i%p]+buf[(i+1)%p])*0.996
    return out
arp_cache={}
for b in range(nb):
    ch=prog[b%4]; seq=[ch[0]+12,ch[1]+12,ch[2]+12,ch[1]+24]
    for s16 in range(16):
        if b<1: continue  # arp enters after first bar
        m=seq[s16%4]
        if m not in arp_cache: arp_cache[m]=pluck(note(m))
        add(arp_cache[m],b*bar+s16*beat/4+(CFG['swing']*beat/4 if s16%2 else 0),CFG['arp'],pan=(0.35 if s16%2 else -0.35))
# drums
for k in range(int(DUR/beat)+1):
    t0=k*beat
    add(kick(),t0,CFG['kick'])
    if k%2==1: add(clap(),t0,CFG['clap'])
    add(hat(),t0+beat/2,0.22,pan=0.2); sw=CFG['swing']*beat/4; add(hat(),t0+beat/4+sw,0.08,pan=-0.2); add(hat(),t0+3*beat/4+sw,0.08,pan=-0.2)
    if CFG['openhat'] and k%4==3: add(hat(True),t0+beat/2,0.12)
# riser-free intro: filter first bar
mix=np.stack([L,R],1)
# fade in/out
fi=int(.3*SR); fo=int(1.5*SR); mix[:fi]*=np.linspace(0,1,fi)[:,None]; mix[-fo:]*=np.linspace(1,0,fo)[:,None]
mix=np.tanh(mix*1.2); mix/=np.max(np.abs(mix))*1.12
wavfile.write(sys.argv[2] if len(sys.argv)>2 else 'track.wav',SR,(mix*32767).astype(np.int16))
