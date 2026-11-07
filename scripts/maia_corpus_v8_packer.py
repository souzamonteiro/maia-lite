#!/usr/bin/env python3
"""Maia Corpus v8: resumable Drive -> local ZIP -> Drive packer (CPU only)."""
import argparse, hashlib, json, os, shutil, sqlite3, sys, time, zipfile
from pathlib import Path

ROOT_REL = 'Colab_LLMs/Maia_Lite_0_3_0_Chat/corpus_v8/runs/acquisition_v8_5_001/pretraining_production_v1_full'
DB_REL = 'recovery_audit_v4/state_operational_v5.sqlite3'
BLOCK_BYTES = 2048 * 4

def digest(path, chunk=4*1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def source_path(root, value, source, split, idx):
    # Catalog path is authoritative; accept paths from the original Colab environment.
    p=Path(value)
    if p.is_absolute():
        s=str(p)
        marker='/pretraining_production_v1_full/'
        if marker in s:
            p=root / s.split(marker,1)[1]
        elif p.exists():
            return p
        else:
            raise ValueError(f'Unexpected absolute catalog path: {value}')
    else:
        if p.parts and p.parts[0]=='shards': p=root/p
        else: p=root/'shards'/p
    if not p.is_file():
        fallback=root/'shards'/source/split/f'{source}_{split}_{idx:06d}.bin'
        if fallback.is_file(): return fallback
        raise FileNotFoundError(str(p))
    return p

def plan(rows, max_bytes):
    packs=[]; current=[]; total=0
    for r in rows:
        size=r[4]*BLOCK_BYTES
        if current and total+size>max_bytes:
            packs.append(current);current=[];total=0
        current.append(r);total+=size
    if current: packs.append(current)
    return packs

def write_json_atomic(path, data):
    tmp=path.with_name(path.name+'.partial')
    with open(tmp,'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,sort_keys=True)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--drive-root',default='/content/drive/MyDrive')
    ap.add_argument('--local-root',default='/content/maia_corpus_pack_work')
    ap.add_argument('--pack-gib',type=float,default=1.0)
    ap.add_argument('--hours',type=float,default=22.0)
    ap.add_argument('--max-packs',type=int,default=0)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.pack_gib<=0 or args.hours<=0: ap.error('pack-gib and hours must be positive')
    root=Path(args.drive_root)/ROOT_REL
    db=root/DB_REL
    dest=root/'packed_zip_v1'
    local=Path(args.local_root)
    if not db.exists(): raise FileNotFoundError(f'Missing catalog: {db}')
    # Query only; never run PRAGMA quick_check on mounted Drive.
    uri=db.as_uri()+'?mode=ro'
    con=sqlite3.connect(uri,uri=True,timeout=30)
    try:
        rows=con.execute('SELECT source,split,shard_index,path,blocks,sha256 FROM shard ORDER BY source,split,shard_index').fetchall()
    finally: con.close()
    assert len(rows)==60612, f'Unexpected shard count: {len(rows)}'
    assert sum(r[4] for r in rows)==3607119, 'Unexpected block count'
    # Fixed grouping by metadata, stable across restarts.
    packs=plan(rows,int(args.pack_gib*(1024**3)))
    print(f'Shards: {len(rows):,}; blocks: {sum(r[4] for r in rows):,}; ZIP packs: {len(packs)}',flush=True)
    if args.dry_run: return
    dest.mkdir(parents=True,exist_ok=True)
    local.mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); completed=0; skipped=0
    for i, group in enumerate(packs):
        if time.monotonic()-started >= args.hours*3600:
            print('Time budget reached. Safe stop before next pack.',flush=True);break
        source,split=group[0][0:2]
        # Pack names deterministic, independent of session.
        name=f'pack_{i:04d}_{source}_{split}.zip'
        remote=dest/name; meta=dest/(name+'.json')
        expected_bytes=sum(r[4]*BLOCK_BYTES for r in group)
        if remote.exists() and meta.exists():
            try:
                info=json.loads(meta.read_text(encoding='utf-8'))
                if (info.get('status')=='complete' and info.get('zip_bytes')==remote.stat().st_size
                    and info.get('raw_bytes')==expected_bytes and info.get('count')==len(group)
                    and info.get('first')==list(group[0][:3]) and info.get('last')==list(group[-1][:3])):
                    skipped+=1
                    print(f'[{i+1}/{len(packs)}] REUSE {name}',flush=True)
                    continue
            except Exception as e: print(f'Cannot reuse {name}: {e}',flush=True)
            raise RuntimeError(f'Incomplete or conflicting destination pack: {name}; inspect before retry')
        if remote.exists() or meta.exists():
            raise RuntimeError(f'Orphaned ZIP/manifest for {name}; inspect before retry')
        free=shutil.disk_usage(local).free
        if free < expected_bytes + 2*(1024**3):
            raise RuntimeError(f'Insufficient local disk for {name}: free={free}, needed~{expected_bytes+2*(1024**3)}')
        temp=local/(name+'.partial');temp.unlink(missing_ok=True)
        print(f'[{i+1}/{len(packs)}] BUILD {name} shards={len(group)} rawGiB={expected_bytes/1024**3:.3f}',flush=True)
        t=time.monotonic()
        try:
            with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
                for j,(src,sp,idx,path,blocks,sha) in enumerate(group,1):
                    p=source_path(root,path,src,sp,idx)
                    size=blocks*BLOCK_BYTES
                    if p.stat().st_size!=size: raise ValueError(f'Bad shard size: {p}')
                    arc=f'{src}/{sp}/{src}_{sp}_{idx:06d}.bin'
                    h=hashlib.sha256()
                    with open(p,'rb') as inp, z.open(arc,'w',force_zip64=True) as out:
                        count=0
                        while True:
                            buf=inp.read(4*1024*1024)
                            if not buf:break
                            out.write(buf);h.update(buf);count+=len(buf)
                    if count!=size or h.hexdigest().lower()!=sha.lower():
                        raise ValueError(f'Shard SHA256/size mismatch: {p}')
                    if j%100==0 or j==len(group):
                        print(f'  {j}/{len(group)} shards | {time.monotonic()-t:.0f}s',flush=True)
            # Verify ZIP central directory and contents integrity locally.
            with zipfile.ZipFile(temp,'r') as z:
                if len(z.infolist())!=len(group) or z.testzip() is not None:
                    raise ValueError('ZIP integrity check failed')
            zhash=digest(temp)
            zsize=temp.stat().st_size
            # Copy completed ZIP to Drive using staging name, then rename.
            staged=dest/(name+'.uploading')
            staged.unlink(missing_ok=True)
            try:
                shutil.copyfile(temp,staged)
                if staged.stat().st_size!=zsize:
                    raise IOError('Remote staged ZIP size mismatch')
                # Avoid rereading remote ZIP (can be very slow on Drive mount).
                os.replace(staged,remote)
            except Exception:
                print(f'Upload failed; staged file may remain: {staged}',flush=True)
                raise
            info={'status':'complete','zip_name':name,'zip_bytes':zsize,'zip_sha256':zhash,
                  'raw_bytes':expected_bytes,'count':len(group),
                  'first':list(group[0][:3]),'last':list(group[-1][:3]),
                  'sha256_verified_per_shard':True,'format':'zip-stored-v1',
                  'catalog_shards':len(rows),'catalog_blocks':sum(r[4] for r in rows)}
            write_json_atomic(meta,info)
            temp.unlink(missing_ok=True)
            completed+=1
            print(f'  SAVED {name} {zsize/1024**3:.3f} GiB sha256={zhash} elapsed={time.monotonic()-t:.1f}s',flush=True)
            if args.max_packs and completed>=args.max_packs:
                print('max-packs reached',flush=True);break
        except Exception:
            print(f'FAILED {name}; local partial preserved for diagnosis: {temp}',flush=True)
            raise
    print(f'DONE this run: completed={completed} reused={skipped}; total packs={len(packs)}',flush=True)
    print('IMPORTANT: remote ZIP SHA256 must be verified after downloading to SSD before training.',flush=True)

if __name__=='__main__':main()
