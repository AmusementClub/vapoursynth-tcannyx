import json, sys, datetime, statistics, bisect
def load_smi(fn):
    t=[];rows=[]
    for l in open(fn):
        p=[x.strip() for x in l.split(',')]
        if len(p)<6: continue
        try: ts=datetime.datetime.strptime(p[0],"%Y/%m/%d %H:%M:%S.%f").timestamp()
        except Exception: continue
        t.append(ts); rows.append(dict(util=float(p[1]),mem=float(p[2]),clk=float(p[3]),temp=float(p[4]),pw=float(p[5]) if p[5].replace('.','').isdigit() else 0))
    return t,rows
def join(jsonl, smi):
    t,rows=load_smi(smi); out=[]
    for l in open(jsonl):
        r=json.loads(l)
        if 'runs' not in r: continue
        utils=[];mems=[];clks=[];temps=[];pws=[]
        for run in r['runs']:
            e=run['wall_epoch_end']; s=e-run['wall']
            i=bisect.bisect_left(t,s); j=bisect.bisect_right(t,e)
            seg=rows[i:j]
            if seg:
                utils.append(statistics.mean(x['util'] for x in seg)); mems.append(max(x['mem'] for x in seg)); clks.append(statistics.mean(x['clk'] for x in seg)); temps.append(max(x['temp'] for x in seg)); pws.append(statistics.mean(x['pw'] for x in seg))
        r['gpu_util_mean']=statistics.median(utils) if utils else None
        r['gpu_mem_max_mib']=max(mems) if mems else None
        r['gpu_clk_mean']=statistics.median(clks) if clks else None
        r['gpu_temp_max']=max(temps) if temps else None
        r['gpu_power_mean']=statistics.median(pws) if pws else None
        out.append(r)
    return out
if __name__=="__main__":
    out=join(sys.argv[1],sys.argv[2])
    for r in out:
        s=r['spec']
        if s['backend'] in('vulkan','cuda'):
            print(s['backend'],s['modes'],s['threads'],'util %.0f%% clk %.0f temp %.0f pw %.0f'%(r['gpu_util_mean'] or -1,r['gpu_clk_mean'] or -1,r['gpu_temp_max'] or -1,r['gpu_power_mean'] or -1), 'fps %.0f'%r['fps_median'])
