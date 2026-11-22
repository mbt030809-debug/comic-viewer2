#!/usr/bin/env python3
"""후처리 시험(예측 tsv → 정리된 tsv): 전면 무늬 합치기 + 반복 확인.
python3 post.py <truth.json(이미지 크기용)> <in.tsv> <out.tsv> [--tile N] [--vote F] [--mode strict|loose]"""
import sys, json, math, collections
def load(path):
    P = collections.defaultdict(list); order = []
    for l in open(path, encoding='utf-8'):
        f = l.rstrip('\n').split('\t')
        if len(f) < 3: continue
        k = (f[0], f[1]); 
        if k not in order: order.append(k)
        if f[2] != 'NONE': P[k].append([int(f[2]), int(f[3]), int(f[4]), int(f[5]), f[6], float(f[7])])
    return P, order

def similar(a, b, tol=.35): return abs(a[2] - b[2]) <= tol * max(a[2], b[2]) + 6 and abs(a[3] - b[3]) <= tol * max(a[3], b[3]) + 6

def consistent(a, ha, b, hb, mode):
    """같은 워터마크로 볼 만한 위치 관계: 위/아래/비율 정렬, 또는 x만 같음(세로 제각각). loose 는 위치 무관"""
    if not similar(a, b): return False
    if mode == 'loose': return True
    dx = abs(a[0] - b[0]); 
    if dx > 24: return False
    top = abs(a[1] - b[1]) <= 24; bot = abs((ha - a[1] - a[3]) - (hb - b[1] - b[3])) <= 24
    rat = abs((a[1] + a[3] / 2) / ha - (b[1] + b[3] / 2) / hb) <= .02
    return top or bot or rat or True   # x가 같으면 free_y 로 인정

if __name__ == '__main__':
    truth, inp, outp = sys.argv[1:4]
    arg = lambda n, d: type(d)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d
    tileN = arg('--tile', 6); vote = arg('--vote', 0.0); mode = arg('--mode', 'strict'); T = {s['id']: s for s in json.load(open(truth))['sites']}
    P, order = load(inp); size = {}
    for s in T.values():
        for im in s['images']: size[(s['id'], im['file'])] = (im['w'], im['h'])
    out = collections.defaultdict(list)
    for k in order:
        w, h = size[k]; bs = P.get(k, [])
        if sum(1 for b in bs if b[4] == 'mark') >= tileN and tileN > 0:   # 비슷한 작은 덩어리가 많으면 전면 무늬
            ms = sorted(b[2] * b[3] for b in bs); med = ms[len(ms) // 2]
            if sum(1 for a in ms if .3 * med <= a <= 3 * med) >= tileN: bs = [[0, 0, w, h, 'tile', 1.0]]
        out[k] = bs
    if vote > 0:
        bysite = collections.defaultdict(list)
        for k in order: bysite[k[0]].append(k)
        for sid, ks in bysite.items():
            for k in ks:
                keep = []
                for b in out[k]:
                    if b[4] == 'tile': keep.append(b); continue
                    others = [o for o in ks if o != k and size[o][0] == size[k][0]]; need = max(1, math.ceil(vote * len(others)))
                    cnt = sum(1 for o in others if any(consistent(b, size[k][1], c, size[o][1], mode) for c in out[o] if c[4] == 'mark'))
                    if cnt >= need: keep.append(b)
                out[k] = keep
    lines = []
    for k in order:
        if not out[k]: lines.append('%s\t%s\tNONE' % k)
        for b in out[k]: lines.append('%s\t%s\t%d\t%d\t%d\t%d\t%s\t%.3f' % (k[0], k[1], *b))
    open(outp, 'w').write('\n'.join(lines) + '\n')
