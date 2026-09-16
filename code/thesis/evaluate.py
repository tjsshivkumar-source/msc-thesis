import os, re, csv, glob, argparse
import numpy as np
from PIL import Image
import cv2
import torch
import lpips
from skimage.metrics import structural_similarity as ssim_fn

DATA_ROOT = '/vol/gpudata/tes25-thesis/data/patent_tests/THESIS_IMGS'
OUT_ROOT  = '/vol/gpudata/tes25-thesis/outputs/patent_baseline'
DEBUG_DIR = os.path.join(OUT_ROOT, 'debug_pairs')
RES = 512
IMG_EXT = ('.jpg', '.jpeg', '.png')

# ---------- image loading + normalization ----------

def load_rgb(path):
    """Load any image, composite alpha onto detected background, return RGB uint8."""
    im = Image.open(path).convert('RGBA') #convert loaded image to RGBA (where A is opacity)
    a = np.array(im, dtype=np.float32)
    rgb, alpha = a[..., :3], a[..., 3:] / 255.0
    rgb = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)  # composite onto white
    return rgb

def normalize(rgb):
    """bg-detect -> content-crop -> square-pad -> resize -> (rgb, gray)."""
    #detect background from corner pixels, alpha-composite to white, crop to content bounding box, 
    #pad square, resize 512, grayscale copy
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) # detect gray bg
    # background from the four corners (renders may be black-bg, drawings white-bg)
    corners = [gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]]
    bg = int(np.median(corners))
    mask = np.abs(gray.astype(np.int16) - bg) > 12 #make a mask that subtracts gray detected bg 
    ys, xs = np.where(mask)
    if len(ys) > 0:
        rgb = rgb[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = rgb.shape[:2]; s = max(h, w)
    pad = np.full((s, s, 3), 255, np.uint8) # pad with white always
    pad[(s-h)//2:(s-h)//2+h, (s-w)//2:(s-w)//2+w] = rgb
    rgb = cv2.resize(pad, (RES, RES), interpolation=cv2.INTER_AREA)
    # if source bg was dark, invert composite to white-bg convention for comparability
    if bg < 128:
        gray_r = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # re-composite: pixels near old bg become white
        bgmask = np.abs(gray_r.astype(np.int16) - bg) <= 12
        rgb = rgb.copy(); rgb[bgmask] = 255
    return rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def edge_image(gray):
    e = cv2.Canny(gray, 50, 150)
    return 255 - e   # black lines on white, matching drawing convention

# ---------- metrics ----------

_loss = None
def lpips_fn():
    global _loss
    if _loss is None:
        _loss = lpips.LPIPS(net='alex')
        if torch.cuda.is_available(): _loss = _loss.cuda()
    return _loss

def to_t(rgb):
    #convert rgb HWC array to tensor
    #rgb / 127.5 - 1.0: Normalizes the pixel values (req for diffuysion models)
    #torch.tensor(..., dtype=torch.float32): Converts the array (usually a NumPy array) into a PyTorch tensor
    #.permute(2, 0, 1): Reorders the dimensions
    #[None]: Adds a new dimension at the very front. This converts the (C, H, W) tensor into a (1, C, H, W) tensor, adding the "Batch" dimension (N=1) required by neural network inputs
    t = torch.tensor(rgb / 127.5 - 1.0, dtype=torch.float32).permute(2, 0, 1)[None]
    return t.cuda() if torch.cuda.is_available() else t

def compare(r_rgb, r_gray, g_rgb, g_gray):
    d  = lpips_fn()(to_t(r_rgb), to_t(g_rgb)).item()
    s  = ssim_fn(r_gray, g_gray)
    #Canny-extract the render's edges first, so a line-drawing-like version of the
    #render is compared against the actual line drawing (fairer across the media gap)
    re = edge_image(r_gray); re3 = np.stack([re]*3, -1)
    de = lpips_fn()(to_t(re3), to_t(g_rgb)).item()
    se = ssim_fn(re, g_gray)
    return d, s, de, se

# ---------- dataset walking ----------

def read_manifest():
    cond = {}
    with open(os.path.join(OUT_ROOT, 'manifest.csv')) as fh:
        for row in csv.DictReader(fh):
            cond[row['item']] = row['cond_view']
    return cond

def gt_views(dm, style):
    """All ground-truth view files for an item; matches _LINE/_PHOTO subfolders."""
    hits = []
    for dirpath, _, files in os.walk(DATA_ROOT):
        base = os.path.basename(dirpath)
        if base == dm or base == f'{dm}_{style}':
            for fn in files:
                if fn.lower().endswith(IMG_EXT):
                    hits.append(os.path.join(dirpath, fn))
    return hits

def view_name(path):
    m = re.search(r'_(PERSP\d*|FRONT|BACK|LEFT|RIGHT|TOP|BOTTOM)\.', os.path.basename(path), re.I)
    return m.group(1).upper() if m else None

# ---------- main ----------

def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cond = read_manifest()
    rows = []
    for vdir in sorted(glob.glob(f'{OUT_ROOT}/*/views')):
        item = os.path.basename(os.path.dirname(vdir))       # DM077749_LINE
        m = re.match(r'(DM\d+)_(\w+)', item)
        dm, style = m.group(1), m.group(2)
        cview = cond.get(item, '')
        cname = view_name('_' + cview) if cview else None    # cond filename -> view token
        for gt_path in gt_views(dm, style):
            vn = view_name(gt_path)
            if vn is None or (cname and vn == cname):        # skip conditioning view
                continue
            g_rgb, g_gray = normalize(load_rgb(gt_path))
            base = vn.rstrip('0123456789')
            if base == 'PERSP':
                candidates = sorted(glob.glob(f'{vdir}/PERSP_*.png'))
                swept = True
            else:
                candidates = [f'{vdir}/{base}.png']
                swept = False
            best = None
            for cpath in candidates:
                if not os.path.exists(cpath): continue
                r_rgb, r_gray = normalize(load_rgb(cpath))
                met = compare(r_rgb, r_gray, g_rgb, g_gray)
                if best is None or met[1] > best[0][1]:
                    best = (met, cpath, r_rgb)
            if best is None:
                print(f'NO RENDER for {item}/{vn}'); continue
            (d, s, de, se), cpath, r_rgb = best
            off = re.search(r'PERSP_([+-]\d+)_([+-]\d+)', os.path.basename(cpath))
            dyaw, dpitch = (int(off.group(1)), int(off.group(2))) if off else (0, 0)
            rows.append([item, dm, style, vn, swept, f'{d:.4f}', f'{s:.4f}',
                         f'{de:.4f}', f'{se:.4f}', dyaw, dpitch])
            # debug pair
            pair = np.concatenate([r_rgb, np.full((RES, 8, 3), 128, np.uint8),
                                   normalize(load_rgb(gt_path))[0]], axis=1)
            Image.fromarray(pair).save(f'{DEBUG_DIR}/{item}_{vn}.png')
            print(f'{item:24s} {vn:8s} lpips={d:.3f} ssim={s:.3f} edge_ssim={se:.3f}'
                  + (f' (sweep {dyaw:+d},{dpitch:+d})' if swept else ''))
    with open(os.path.join(OUT_ROOT, 'eval_results.csv'), 'w', newline='') as fh:
        csv.writer(fh).writerows(
            [['item','dm','style','view','swept','lpips','ssim','lpips_edge','ssim_edge','dyaw','dpitch']] + rows)
    print(f'\n{len(rows)} comparisons -> eval_results.csv')

def selftest():
    """GT-vs-GT: same image ~ perfect; cross-view ~ bad. No renders needed."""
    #used to test whether the SSIM/LPIPS metrics work on pre-rendered baseline views in the patent_baseline dir
    items = sorted(glob.glob(f'{DATA_ROOT}/**/*.jpg', recursive=True))[:1]
    p = items[0]
    g1 = normalize(load_rgb(p)); g2 = normalize(load_rgb(p))
    d, s, de, se = compare(g1[0], g1[1], g2[0], g2[1])
    print(f'SAME-IMAGE   lpips={d:.4f} (expect ~0)  ssim={s:.4f} (expect ~1)')
    dirp = os.path.dirname(p)
    others = [f for f in glob.glob(f'{dirp}/*.jpg') if f != p]
    if others:
        g3 = normalize(load_rgb(others[0]))
        d, s, de, se = compare(g1[0], g1[1], g3[0], g3[1])
        print(f'CROSS-VIEW   lpips={d:.4f} (expect >>0) ssim={s:.4f} (expect <<1)')
    os.makedirs(DEBUG_DIR, exist_ok=True)
    pair = np.concatenate([g1[0], np.full((RES, 8, 3), 128, np.uint8), g3[0]], axis=1)
    Image.fromarray(pair).save(f'{DEBUG_DIR}/SELFTEST_pair.png')
    print(f'Eyeball {DEBUG_DIR}/SELFTEST_pair.png for normalization sanity')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest() if a.selftest else main()
    