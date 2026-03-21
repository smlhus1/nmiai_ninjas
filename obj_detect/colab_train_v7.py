"""
NorgesGruppen Object Detection — v7 Training Pipeline
Reproducible end-to-end: setup → corrected fine-tune → pseudo-labeling → model soup → final models

Run in Google Colab with A100 GPU (80GB VRAM recommended).
Each step is independent and saves to Drive. If a step crashes, restart from that step.

Requirements:
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    pip install ultralytics==8.1.0 ensemble-boxes albumentations

Drive structure:
    nmiai_weights/
    ├── obj_detect/NM_NGD_coco_dataset.zip
    ├── obj_detect/NM_NGD_product_images.zip
    ├── v6_m1_oiv7_l_1280_cp.pt  (base model from v6 training)
    └── (v7 outputs saved here)
"""

# ============================================================
# STEP 0: Setup
# ============================================================
def step0_setup():
    """Install deps, mount Drive, build datasets."""
    import subprocess
    subprocess.run(['pip', 'install', 'torch==2.6.0', 'torchvision==0.21.0',
                    '--index-url', 'https://download.pytorch.org/whl/cu124', '-q'], check=True)
    subprocess.run(['pip', 'install', 'ultralytics==8.1.0', 'ensemble-boxes', 'albumentations', '-q'], check=True)

    import os, torch, zipfile, json, shutil, random, re
    import numpy as np
    from pathlib import Path
    from collections import defaultdict

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE'] = 'disabled'

    import torch.serialization
    if not getattr(torch, '_patched_load', False):
        _real_load = torch.serialization.load
        torch.load = lambda *args, **kwargs: _real_load(*args, **{**kwargs, 'weights_only': False})
        torch._patched_load = True

    if not hasattr(np, 'trapz'): np.trapz = np.trapezoid

    from google.colab import drive
    drive.mount('/content/drive')

    print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")

    SAVE_DIR = Path('/content/drive/MyDrive/nmiai_weights')

    # Extract data
    src_zip = SAVE_DIR / 'obj_detect'
    for zf, dest in [('NM_NGD_coco_dataset.zip','coco_dataset'),('NM_NGD_product_images.zip','product_images')]:
        if not Path(dest).exists():
            with zipfile.ZipFile(src_zip/zf) as z: z.extractall(dest)

    # Parse annotations
    ann_data = json.loads(Path('coco_dataset/train/annotations.json').read_text())
    images = {img['id']: img for img in ann_data['images']}
    img_anns = defaultdict(list)
    for ann in ann_data['annotations']: img_anns[ann['image_id']].append(ann)
    nc = len(ann_data['categories'])

    # Train/val split (seed 42, stratified by annotation count)
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

    # Build dataset_mc (multi-class, no augmentation)
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

    # Build copy-paste dataset
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

    print(f"MC: {len(list((mc/'train'/'images').glob('*.jpg')))} train, {len(list((mc/'val'/'images').glob('*.jpg')))} val")
    print(f"CP: {len(list((cp/'train'/'images').glob('*.jpg')))} train")
    print("SETUP DONE")


# ============================================================
# STEP 1: Fine-tune on clean data (low LR, frozen backbone)
# ============================================================
def step1_finetune():
    """Fine-tune M1 on dataset_mc with very low LR. No corrected annotations in this dataset."""
    import torch, gc, shutil, time
    from pathlib import Path
    from ultralytics import YOLO

    gc.collect(); torch.cuda.empty_cache()
    SAVE_DIR = Path('/content/drive/MyDrive/nmiai_weights')

    base = str(SAVE_DIR / 'v6_m1_oiv7_l_1280_cp.pt')
    t0 = time.time()
    m = YOLO(base)
    r = m.train(
        data='dataset_mc/dataset.yaml',
        epochs=30, imgsz=1280, batch=16, device=0, amp=True,
        project='runs', name='v7_finetune', exist_ok=True, patience=10,
        optimizer='AdamW', lr0=0.00005, lrf=0.01, warmup_epochs=2,
        cos_lr=True, label_smoothing=0.0, freeze=15,
        mosaic=0.0, mixup=0.0, close_mosaic=0,
        scale=0.2, fliplr=0.5,
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.2,
        workers=4,
    )
    mAP = r.results_dict['metrics/mAP50(B)']
    print(f"\n>>> Fine-tune DONE in {(time.time()-t0)/60:.1f}min — mAP@0.5: {mAP:.4f}")
    shutil.copy2('runs/v7_finetune/weights/best.pt', SAVE_DIR / 'v7_corrected_ft.pt')
    print("Saved to Drive!")
    del m, r; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# STEP 2: Pseudo-labeling
# ============================================================
def step2_pseudo_labeling():
    """Generate pseudo-labels from teacher model on augmented images, build combined dataset."""
    import torch, gc, json, random
    import numpy as np
    from pathlib import Path
    from PIL import Image
    from ultralytics import YOLO
    import albumentations as A
    import shutil

    gc.collect(); torch.cuda.empty_cache()
    SAVE_DIR = Path('/content/drive/MyDrive/nmiai_weights')

    # Use best available model as teacher
    corr_pt = SAVE_DIR / 'v7_corrected_ft.pt'
    base_pt = SAVE_DIR / 'v6_m1_oiv7_l_1280_cp.pt'
    teacher = YOLO(str(corr_pt if corr_pt.exists() else base_pt))
    print(f"Teacher: {corr_pt.name if corr_pt.exists() else base_pt.name}")

    heavy_aug = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.8),
        A.CLAHE(clip_limit=4.0, p=0.3),
    ])

    pseudo_dir = Path('dataset_pseudo')
    (pseudo_dir/'train'/'images').mkdir(parents=True, exist_ok=True)
    (pseudo_dir/'train'/'labels').mkdir(parents=True, exist_ok=True)

    src_imgs = sorted(Path('dataset_mc/train/images').glob('*.jpg'))
    print(f"Generating augmented images from {len(src_imgs)} originals...")

    random.seed(123)
    n_generated = 0
    for img_path in src_imgs:
        img = np.array(Image.open(img_path))
        for i in range(3):
            aug_img = heavy_aug(image=img)['image']
            aug_name = f"{img_path.stem}_pseudo{i}.jpg"
            Image.fromarray(aug_img).save(pseudo_dir/'train'/'images'/aug_name, quality=90)
            n_generated += 1
    print(f"Generated {n_generated} augmented images")

    # Generate pseudo-labels with high confidence
    print("Generating pseudo-labels (conf=0.5)...")
    aug_imgs = sorted((pseudo_dir/'train'/'images').glob('*.jpg'))
    labeled = 0
    for img_path in aug_imgs:
        results = teacher(str(img_path), device='cuda', verbose=False,
                         conf=0.5, iou=0.5, max_det=500, imgsz=1280)
        lines = []
        for r in results:
            if r.boxes is None: continue
            img_h, img_w = r.orig_shape
            for i in range(len(r.boxes)):
                x1,y1,x2,y2 = r.boxes.xyxy[i].tolist()
                cls = int(r.boxes.cls[i].item())
                lines.append(f"{cls} {(x1+x2)/2/img_w:.6f} {(y1+y2)/2/img_h:.6f} {(x2-x1)/img_w:.6f} {(y2-y1)/img_h:.6f}")
        if len(lines) >= 10:
            (pseudo_dir/'train'/'labels'/f"{img_path.stem}.txt").write_text('\n'.join(lines))
            labeled += 1
    print(f"Pseudo-labeled {labeled}/{n_generated} images")

    # Build combined dataset: original (2x weight) + pseudo
    combined = Path('dataset_combined')
    if combined.exists(): shutil.rmtree(combined)
    shutil.copytree('dataset_mc/val', str(combined/'val'))
    (combined/'train'/'images').mkdir(parents=True, exist_ok=True)
    (combined/'train'/'labels').mkdir(parents=True, exist_ok=True)

    # Originals
    for f in Path('dataset_mc/train/images').glob('*.jpg'):
        shutil.copy2(f, combined/'train'/'images'/f.name)
    for f in Path('dataset_mc/train/labels').glob('*.txt'):
        shutil.copy2(f, combined/'train'/'labels'/f.name)
    # Originals 2x
    for f in Path('dataset_mc/train/images').glob('*.jpg'):
        shutil.copy2(f, combined/'train'/'images'/f"dup_{f.name}")
    for f in Path('dataset_mc/train/labels').glob('*.txt'):
        shutil.copy2(f, combined/'train'/'labels'/f"dup_{f.name}")
    # Pseudo-labeled
    for f in (pseudo_dir/'train'/'images').glob('*.jpg'):
        label = pseudo_dir/'train'/'labels'/f"{f.stem}.txt"
        if label.exists():
            shutil.copy2(f, combined/'train'/'images'/f.name)
            shutil.copy2(label, combined/'train'/'labels'/f"{f.stem}.txt")

    ann_data = json.loads(Path('coco_dataset/train/annotations.json').read_text())
    nc = len(ann_data['categories'])
    names = '\n'.join(f'  {c["id"]}: "{c["name"]}"' for c in ann_data['categories'])
    (combined/'dataset.yaml').write_text(f"path: {combined.resolve()}\ntrain: train/images\nval: val/images\nnc: {nc}\nnames:\n{names}\n")

    n_train = len(list((combined/'train'/'images').glob('*.jpg')))
    print(f"Combined dataset: {n_train} train images")
    print("PSEUDO-LABELING DONE")
    del teacher; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# STEP 3: Model soup (3 seeds, average weights)
# ============================================================
def step3_model_soup():
    """Train 3 models with different seeds on combined data, average weights."""
    import torch, gc, shutil, time
    from pathlib import Path
    from collections import OrderedDict
    from ultralytics import YOLO

    gc.collect(); torch.cuda.empty_cache()
    SAVE_DIR = Path('/content/drive/MyDrive/nmiai_weights')

    seeds = [0, 42, 123]
    best_pts = []

    for seed in seeds:
        name = f'v7_soup_seed{seed}'
        drive_pt = SAVE_DIR / f'{name}.pt'
        if drive_pt.exists():
            print(f"\nSkip {name} — already on Drive")
            best_pts.append(str(drive_pt))
            continue

        print(f"\n{'='*60}")
        print(f"{name} (seed={seed})")
        print(f"{'='*60}")
        gc.collect(); torch.cuda.empty_cache()

        t0 = time.time()
        m = YOLO('yolov8l-oiv7.pt')
        r = m.train(
            data='dataset_combined/dataset.yaml',
            epochs=150, imgsz=1280, batch=16, device=0, amp=True,
            project='runs', name=name, exist_ok=True, patience=0,
            seed=seed,
            optimizer='AdamW', lr0=0.001, lrf=0.01, warmup_epochs=5,
            cos_lr=True, label_smoothing=0.1, freeze=10,
            mosaic=1.0, mixup=0.15, erasing=0.3, scale=0.5, fliplr=0.5,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            workers=4,
        )
        mAP = r.results_dict['metrics/mAP50(B)']
        print(f"\n>>> {name} DONE in {(time.time()-t0)/60:.1f}min — mAP@0.5: {mAP:.4f}")
        shutil.copy2(f'runs/{name}/weights/best.pt', drive_pt)
        best_pts.append(str(drive_pt))
        print("Saved to Drive!")
        del m, r; gc.collect(); torch.cuda.empty_cache()

    # Average weights
    print(f"\n{'='*60}")
    print(f"Averaging {len(best_pts)} models...")
    print(f"{'='*60}")
    avg_state = None
    n = 0
    for pt_path in best_pts:
        ckpt = torch.load(pt_path, map_location='cpu')
        model_obj = ckpt.get('ema', ckpt.get('model'))
        state = model_obj.float().state_dict()
        if avg_state is None:
            avg_state = OrderedDict((k, v.clone()) for k, v in state.items())
        else:
            for k in avg_state: avg_state[k] += state[k]
        n += 1
    for k in avg_state: avg_state[k] /= n

    base_ckpt = torch.load(best_pts[0], map_location='cpu')
    base_model = base_ckpt.get('ema', base_ckpt.get('model'))
    base_model.float().load_state_dict(avg_state)
    soup_ckpt = {'model': base_model, 'train_args': base_ckpt.get('train_args', {})}
    soup_path = SAVE_DIR / 'v7_soup_avg.pt'
    torch.save(soup_ckpt, soup_path)
    print(f"Soup model saved: {soup_path.stat().st_size/1e6:.1f}MB")
    print("MODEL SOUP DONE")


# ============================================================
# STEP 4: Final models on 100% data
# ============================================================
def step4_final_models():
    """Train 3 final models on ALL data (no val holdout) + pseudo-labels."""
    import torch, gc, shutil, time, json, zipfile
    import numpy as np
    from pathlib import Path
    from collections import defaultdict
    from ultralytics import YOLO

    gc.collect(); torch.cuda.empty_cache()
    SAVE_DIR = Path('/content/drive/MyDrive/nmiai_weights')

    # Build 100% dataset
    full = Path('dataset_full')
    if not full.exists():
        ann_data = json.loads(Path('coco_dataset/train/annotations.json').read_text())
        images_info = {img['id']: img for img in ann_data['images']}
        img_anns = defaultdict(list)
        for ann in ann_data['annotations']: img_anns[ann['image_id']].append(ann)
        nc = len(ann_data['categories'])

        def coco_to_yolo(bbox, w, h):
            x,y,bw,bh = bbox
            return (max(0,min(1,(x+bw/2)/w)),max(0,min(1,(y+bh/2)/h)),max(0,min(1,bw/w)),max(0,min(1,bh/h)))

        src = Path('coco_dataset/train/images')
        for split in ['train', 'val']:
            (full/split/'images').mkdir(parents=True, exist_ok=True)
            (full/split/'labels').mkdir(parents=True, exist_ok=True)
            for img_id in images_info:
                info = images_info[img_id]; fname = info['file_name']; w,h = info['width'],info['height']
                for ext in [fname, fname.replace('.jpeg','.jpg'), fname.replace('.jpg','.jpeg')]:
                    s = src/ext
                    if s.exists(): shutil.copy2(s, full/split/'images'/fname.replace('.jpeg','.jpg')); break
                lines = [f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                         for ann in img_anns.get(img_id,[])
                         for cx,cy,nw,nh in [coco_to_yolo(ann['bbox'],w,h)] if nw>0 and nh>0]
                (full/split/'labels'/f"{Path(fname).stem}.txt").write_text('\n'.join(lines))

        # Add pseudo-labeled images
        pseudo_imgs = Path('dataset_pseudo/train/images')
        pseudo_lbls = Path('dataset_pseudo/train/labels')
        if pseudo_imgs.exists():
            for f in pseudo_imgs.glob('*.jpg'):
                lbl = pseudo_lbls / f"{f.stem}.txt"
                if lbl.exists():
                    shutil.copy2(f, full/'train'/'images'/f.name)
                    shutil.copy2(lbl, full/'train'/'labels'/f"{f.stem}.txt")

        names = '\n'.join(f'  {c["id"]}: "{c["name"]}"' for c in ann_data['categories'])
        (full/'dataset.yaml').write_text(f"path: {full.resolve()}\ntrain: train/images\nval: val/images\nnc: {nc}\nnames:\n{names}\n")

    n_train = len(list((full/'train'/'images').glob('*.jpg')))
    print(f"Full dataset: {n_train} train images")

    for name, imgsz, batch, model_file in [
        ('v7_final_l_1280', 1280, 16, 'yolov8l-oiv7.pt'),
        ('v7_final_x_640', 640, 32, 'yolov8x-oiv7.pt'),
        ('v7_final_l_640', 640, 32, 'yolov8l-oiv7.pt'),
    ]:
        drive_pt = SAVE_DIR / f'{name}.pt'
        if drive_pt.exists():
            print(f"\nSkip {name} — already on Drive")
            continue

        print(f"\n{'='*60}")
        print(f"{name} (imgsz={imgsz}, batch={batch})")
        print(f"{'='*60}")
        gc.collect(); torch.cuda.empty_cache()

        t0 = time.time()
        m = YOLO(model_file)
        r = m.train(
            data='dataset_full/dataset.yaml',
            epochs=150, imgsz=imgsz, batch=batch, device=0, amp=True,
            project='runs', name=name, exist_ok=True, patience=0,
            optimizer='AdamW', lr0=0.001, lrf=0.01, warmup_epochs=5,
            cos_lr=True, label_smoothing=0.1, freeze=10,
            mosaic=1.0, mixup=0.15, erasing=0.3, scale=0.5, fliplr=0.5,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            workers=4,
        )
        elapsed = (time.time()-t0)/60
        print(f"\n>>> {name} DONE in {elapsed:.1f}min")
        shutil.copy2(f'runs/{name}/weights/best.pt', drive_pt)
        print("Saved to Drive!")
        del m, r; gc.collect(); torch.cuda.empty_cache()

    print("FINAL MODELS DONE")


# ============================================================
# STEP 5: Build submission
# ============================================================
def step5_build_submission():
    """Strip weights, write run.py, build submission zip."""
    import torch, shutil, zipfile
    from pathlib import Path

    SAVE_DIR = Path('/content/drive/MyDrive/nmiai_weights')

    # Pick best 3 models
    candidates = [
        ('v7_soup_avg.pt', 1280, 'detect_l.pt'),
        ('v7_final_x_640.pt', 640, 'detect_x.pt'),
        ('v7_final_l_640.pt', 640, 'detect_l2.pt'),
    ]
    fallbacks = {
        'detect_l.pt': 'v6_m1_oiv7_l_1280_cp.pt',
        'detect_x.pt': 'v6_m2_oiv7_x_640_cp.pt',
        'detect_l2.pt': 'v6_m3_oiv7_l_640.pt',
    }

    sub = Path('/content/submission_v7')
    if sub.exists(): shutil.rmtree(sub)
    sub.mkdir()

    for src_name, imgsz, dst_name in candidates:
        src = SAVE_DIR / src_name
        if not src.exists():
            src = SAVE_DIR / fallbacks[dst_name]
        if not src.exists():
            print(f"  {dst_name}: MISSING!")
            continue

        ckpt = torch.load(str(src), map_location='cpu')
        stripped = {'model': ckpt.get('ema', ckpt.get('model')), 'train_args': ckpt.get('train_args', {})}
        dst = sub / dst_name
        torch.save(stripped, dst)
        print(f"  {dst_name}: {src.stat().st_size/1e6:.1f}MB -> {dst.stat().st_size/1e6:.1f}MB")

    # Write run.py (see obj_detect/run_submission_v6.py for latest version)
    from pathlib import Path as P
    run_src = P('/content/drive/MyDrive/nmiai_weights/submission_v6_balanced.zip')
    if run_src.exists():
        import zipfile as zf
        with zf.ZipFile(run_src) as z:
            run_py = z.read('run.py').decode()
        (sub / 'run.py').write_text(run_py)
    else:
        print("WARNING: No run.py template found! Copy from obj_detect/run_submission_v6.py")

    # Verify
    print("\nSubmission contents:")
    total = 0
    for f in sorted(sub.iterdir()):
        sz = f.stat().st_size; total += sz
        print(f"  {f.name}: {sz/1e6:.1f}MB")
    print(f"Total: {total/1e6:.1f}MB / 420MB")

    if total <= 420e6:
        zip_path = Path('/content/submission_v7.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in sub.iterdir(): zf.write(f, f.name)
        print(f"\nsubmission_v7.zip: {zip_path.stat().st_size/1e6:.1f}MB")
        shutil.copy2(zip_path, SAVE_DIR / 'submission_v7.zip')
        print("Copied to Drive! READY TO SUBMIT")
    else:
        print("OVER 420MB LIMIT!")


# ============================================================
# Run all steps
# ============================================================
if __name__ == '__main__':
    step0_setup()
    step1_finetune()
    step2_pseudo_labeling()
    step3_model_soup()
    step4_final_models()
    step5_build_submission()
