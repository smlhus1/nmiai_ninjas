"""
RETRAIN WITH EXACT SANDBOX VERSIONS: torch 2.6.0 + ultralytics 8.1.0

Run in Google Colab with A100 GPU.
Trains 3 OIV7 models for ensemble, saves to Drive.

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics==8.1.0 ensemble-boxes
"""
import subprocess
subprocess.run(['pip', 'install', 'torch==2.6.0', 'torchvision==0.21.0',
                '--index-url', 'https://download.pytorch.org/whl/cu124', '-q'], check=True)
subprocess.run(['pip', 'install', 'ultralytics==8.1.0', '-q'], check=True)
subprocess.run(['pip', 'install', 'ensemble-boxes', '-q'], check=True)

# Fix torch.load weights_only for ultralytics 8.1.0 + PyTorch 2.6
import torch
_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, 'weights_only': False})

print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")

import ultralytics
print(f"ultralytics: {ultralytics.__version__}")
assert torch.__version__.startswith('2.6')
assert ultralytics.__version__ == '8.1.0'
print("✅ Exact sandbox versions!")

from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import shutil, json, random, os, time, zipfile, numpy as np, re
from collections import defaultdict

os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
SAVE_DIR = Path('/content/drive/MyDrive/nmiai_weights')

# Data
src_zip = SAVE_DIR / 'obj_detect'
for zf, dest in [('NM_NGD_coco_dataset.zip','coco_dataset'),('NM_NGD_product_images.zip','product_images')]:
    if not Path(dest).exists():
        with zipfile.ZipFile(src_zip/zf) as z: z.extractall(dest)

ann_data = json.loads(Path('coco_dataset/train/annotations.json').read_text())
images = {img['id']: img for img in ann_data['images']}
img_anns = defaultdict(list)
for ann in ann_data['annotations']: img_anns[ann['image_id']].append(ann)
nc = len(ann_data['categories'])

random.seed(42)
bins = defaultdict(list)
for img_id in sorted(images.keys()):
    count = len(img_anns.get(img_id, []))
    bk = 0 if count==0 else 1 if count<=30 else 2 if count<=80 else 3 if count<=150 else 4
    bins[bk].append(img_id)
train_ids, val_ids = [], []
for bk in sorted(bins.keys()):
    bi = bins[bk]; random.shuffle(bi)
    n_val = max(1, int(len(bi)*0.2))
    val_ids.extend(bi[:n_val]); train_ids.extend(bi[n_val:])

def coco_to_yolo(bbox, w, h):
    x,y,bw,bh = bbox
    return (max(0,min(1,(x+bw/2)/w)),max(0,min(1,(y+bh/2)/h)),max(0,min(1,bw/w)),max(0,min(1,bh/h)))

mc = Path('dataset_mc')
if not mc.exists():
    src = Path('coco_dataset/train/images')
    for split, ids in [('train',train_ids),('val',val_ids)]:
        (mc/split/'images').mkdir(parents=True, exist_ok=True)
        (mc/split/'labels').mkdir(parents=True, exist_ok=True)
        for img_id in ids:
            info = images[img_id]; fname = info['file_name']; w,h = info['width'],info['height']
            for ext in [fname, fname.replace('.jpeg','.jpg'), fname.replace('.jpg','.jpeg')]:
                s = src/ext
                if s.exists(): shutil.copy2(s, mc/split/'images'/fname.replace('.jpeg','.jpg')); break
            lines = [f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                     for ann in img_anns.get(img_id,[])
                     for cx,cy,nw,nh in [coco_to_yolo(ann['bbox'],w,h)] if nw>0 and nh>0]
            (mc/split/'labels'/f"{Path(fname).stem}.txt").write_text('\n'.join(lines))
    names = '\n'.join(f'  {c["id"]}: "{c["name"]}"' for c in ann_data['categories'])
    (mc/'dataset.yaml').write_text(f"path: {mc.resolve()}\ntrain: train/images\nval: val/images\nnc: {nc}\nnames:\n{names}\n")

cp = Path('dataset_cp')
if not cp.exists():
    meta = json.loads(Path('product_images/metadata.json').read_text())
    def normalize(s): return re.sub(r'\s+', ' ', s.strip().upper())
    cat_by_norm = {normalize(c['name']): c['id'] for c in ann_data['categories']}
    cat_to_refs = {}
    for prod in meta['products']:
        if not prod['has_images']: continue
        pname = normalize(prod['product_name'])
        if pname not in cat_by_norm: continue
        cat_id = cat_by_norm[pname]; code = prod['product_code']
        refs = list(Path(f'product_images/{code}').glob('*.jpg')) + list(Path(f'product_images/{code}').glob('*.jpeg'))
        if refs: cat_to_refs[cat_id] = refs
    shutil.copytree(str(mc/'val'), str(cp/'val'))
    shutil.copytree(str(mc/'train'), str(cp/'train'))
    from PIL import Image
    cat_ids_list = list(cat_to_refs.keys())
    for img_path in sorted((mc/'train'/'images').glob('*.jpg')):
        for aug_idx in range(3):
            bg = Image.open(img_path).convert('RGB'); bg_w, bg_h = bg.size
            existing = (mc/'train'/'labels'/f"{img_path.stem}.txt").read_text().strip().split('\n')
            new_labels = list(existing)
            for _ in range(random.randint(3, 10)):
                cat_id = random.choice(cat_ids_list)
                ref_path = random.choice(cat_to_refs[cat_id])
                try: ref = Image.open(ref_path).convert('RGB')
                except: continue
                scale = random.uniform(0.15, 0.6)
                rw, rh = max(10,int(ref.width*scale)), max(10,int(ref.height*scale))
                mx, my = bg_w-rw-1, bg_h-rh-1
                if mx<=0 or my<=0: continue
                x, y = random.randint(0,mx), random.randint(0,my)
                bg.paste(ref.resize((rw,rh), Image.LANCZOS), (x,y))
                new_labels.append(f"{cat_id} {(x+rw/2)/bg_w:.6f} {(y+rh/2)/bg_h:.6f} {rw/bg_w:.6f} {rh/bg_h:.6f}")
            bg.save(cp/'train'/'images'/f"{img_path.stem}_cp{aug_idx}.jpg", quality=90)
            (cp/'train'/'labels'/f"{img_path.stem}_cp{aug_idx}.txt").write_text('\n'.join(new_labels))
    (cp/'dataset.yaml').write_text(f"path: {cp.resolve()}\ntrain: train/images\nval: val/images\nnc: {nc}\nnames:\n" +
        '\n'.join(f'  {c["id"]}: "{c["name"]}"' for c in ann_data['categories']) + '\n')

print(f"MC: {len(list((mc/'train'/'images').glob('*.jpg')))} train")
print(f"CP: {len(list((cp/'train'/'images').glob('*.jpg')))} train")

# Train 3 OIV7 models
from ultralytics import YOLO

for name, data, imgsz, batch, epochs, model_file in [
    ('v6_m1_oiv7_l_1280_cp', 'dataset_cp/dataset.yaml', 1280, 4, 200, 'yolov8l-oiv7.pt'),
    ('v6_m2_oiv7_x_640_cp', 'dataset_cp/dataset.yaml', 640, 8, 150, 'yolov8x-oiv7.pt'),
    ('v6_m3_oiv7_l_640', 'dataset_mc/dataset.yaml', 640, 16, 150, 'yolov8l-oiv7.pt'),
]:
    print(f"\n{'='*60}")
    print(f"{name} (imgsz={imgsz}, {epochs}ep)")
    print(f"{'='*60}")
    t0 = time.time()
    m = YOLO(model_file)
    r = m.train(
        data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=0, amp=True,
        project='runs', name=name, exist_ok=True, patience=0,
        optimizer='AdamW', lr0=0.001, lrf=0.01, warmup_epochs=5,
        cos_lr=True, label_smoothing=0.1, freeze=10,
        mosaic=1.0, mixup=0.15, erasing=0.3, scale=0.5, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    )
    mAP = r.results_dict['metrics/mAP50(B)']
    print(f"\n>>> {name} DONE in {(time.time()-t0)/60:.1f}min — mAP@0.5: {mAP:.4f}")
    shutil.copy2(f'runs/{name}/weights/best.pt', SAVE_DIR / f'{name}.pt')
    print(f"Saved to Drive!")

print(f"\n{'='*60}")
print("ALL DONE! torch 2.6.0 .pt files on Drive.")
print(f"{'='*60}")
