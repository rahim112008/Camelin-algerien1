"""
🧬 Universal Livestock Genomics Pipeline v24.0 (Bovin & Camelin) — COMPLET
"""
import gzip, io, warnings, zipfile, gc, hashlib, json, time, re
from collections import Counter
from io import BytesIO
from urllib.parse import quote
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list, dendrogram
from scipy.spatial.distance import squareform
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS as SklearnMDS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

try:
    from Bio import Align
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="🧬 Helixa | Analyses Génomiques", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")

DEMO = str(st.query_params.get("demo", "")) == "1"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, div { font-family: 'Inter', sans-serif !important; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Space Grotesk', sans-serif !important; color: #e8eaf6 !important; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #04050d !important;
        background-image: radial-gradient(900px 500px at 85% -10%, rgba(138,43,226,.16), transparent 60%), radial-gradient(800px 500px at -10% 30%, rgba(0,191,255,.10), transparent 60%) !important;
        color: #e8eaf6 !important;
    }
    [data-testid="stSidebar"] { background-color: #080a18 !important; border-right: 1px solid rgba(255,255,255,0.1) !important; }
    .stButton>button { background: linear-gradient(110deg, #8a2be2 0%, #00bfff 50%, #ff69b4 100%) !important; color: white !important; border: none !important; border-radius: 999px !important; font-weight: 600 !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} [data-testid="stToolbar"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SPECIES CONFIG
# ============================================================
SPECIES_CONFIG = {
    "Camelin (Dromadaire)": {
        "taxon": "9838", "vep_species": "camelus_dromedarius", "prefix": "camel",
        "autosomes": [str(i) for i in range(1, 37)],
        "ars": {"1": 124.7, "2": 123.8, "3": 120.7, "4": 68.3, "5": 97.5, "6": 97.5, "7": 91.2, "8": 89.4, "9": 86.7, "10": 84.1, "11": 81.5, "12": 78.9, "13": 76.3, "14": 73.8, "15": 71.2, "16": 68.6, "17": 66.1, "18": 63.5, "19": 60.9, "20": 58.4, "21": 55.8, "22": 53.2, "23": 50.7, "24": 48.1, "25": 45.5, "26": 43.0, "27": 40.4, "28": 37.8, "29": 35.3, "30": 32.7, "31": 30.1, "32": 27.6, "33": 25.0, "34": 22.4, "35": 19.9, "36": 17.3, "X": 140.0, "Y": 45.0, "MT": 16.6},
        "groupes": {"Production laitière": {"HAUTE_LAITIERE": ["MOJAHEEM","MAGHATEER","WADAH","AWARIK"], "FAIBLE_LAITIERE": ["BIKANERI","JAISALMERI"]}, "Sous-espèce": {"DROMADAIRE": ["MOJAHEEM","MAGHATEER","WADAH","BIKANERI"], "BACTRIANE": ["KAZAKH_BACTRIAN","ARVANA"]}}
    },
    "Bovin (Taurin/Zébu)": {
        "taxon": "9913", "vep_species": "bos_taurus", "prefix": "bovine",
        "autosomes": [str(i) for i in range(1, 30)],
        "ars": {"1":158.5,"2":136.2,"3":121.0,"4":120.0,"5":120.1,"6":117.8,"7":110.7,"8":113.4,"9":105.7,"10":103.3,"11":107.3,"12":91.2,"13":84.2,"14":84.6,"15":85.2,"16":81.7,"17":75.2,"18":66.1,"19":64.1,"20":72.0,"21":71.6,"22":61.4,"23":52.5,"24":62.4,"25":43.0,"26":51.7,"27":46.8,"28":46.3,"29":51.3,"X":148.8,"Y":57.8,"MT":16.3},
        "groupes": {"Production": {"LAIT": ["NAR","NBD","PTP","PRS"], "VIANDE": ["LJR","SHO","YBS"]}, "Sous-espèce": {"ZEBU": ["PTP","PRS"], "TAURUS": ["LJR","SHO","NAR","NBD"]}}
    }
}

def make_group_map(g): return {r.upper(): grp for grp, races in g.items() for r in races}
def apply_group_mapping(ind_df, gm, default="AUTRE"):
    df = ind_df.copy(); df["FID"] = df["FID"].astype(str).str.upper(); df["FID_GROUPE"] = df["FID"].map(gm).fillna(default); return df

# ============================================================
# UTILS & CACHE
# ============================================================
def _ha(x):
    if not isinstance(x, np.ndarray) or x.size == 0: return "empty"
    if x.dtype.kind in ("U","S","O"):
        h = hashlib.blake2b("|".join(map(str, x.ravel())).encode("utf-8","replace"), digest_size=16); return f"a|{x.shape}|{h.hexdigest()}"
    return f"{x.shape}|{x.dtype}|{hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=16).hexdigest()}"

def _hn(x):
    try: return _ha(np.asarray(x))
    except Exception: return f"{type(x).__name__}"

HF = {np.ndarray: _ha, pd.Series: _hn, pd.Index: _hn}
for _n in ("ArrowStringArray","StringArray","IntegerArray","FloatingArray","BooleanArray","NumpyExtensionArray","PandasArray","DatetimeArray","TimedeltaArray","PeriodArray","IntervalArray","Categorical"):
    _c = getattr(pd.arrays, _n, None) or getattr(pd.core.arrays, _n, None)
    if _c: HF[_c] = _hn

def cache_data(func=None, **kw):
    def _d(f): return st.cache_data(show_spinner=False, hash_funcs=HF, **kw)(f)
    return _d if func is None else _d(func)

def impute_mean(gt):
    g = gt.astype(np.float32, copy=True); cm = np.nanmean(g, 0); cm = np.where(np.isnan(cm), 0.0, cm); nm = np.isnan(g)
    if nm.any(): g[nm] = np.take(cm, np.where(nm)[1])
    return g

def _std_geno(gt):
    X = impute_mean(gt); X = X - X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0; return X / sd

def _autw(ars, autosomes):
    L = np.array([ars[c] for c in autosomes], dtype=float); return L / L.sum()

def _chk(c):
    s = str(c).upper().replace("CHROMOSOME","").replace("CHR","")
    if s.isdigit(): return (0, int(s), "")
    sp = {"X":100,"Y":101,"MT":102,"M":102}
    return (1, sp[s], "") if s in sp else (2, 0, s)

def _chk_num(c):
    t = _chk(c); return t[0]*1000 + t[1]

def sort_chr_bp(df, bp="BP"):
    return df.assign(_k=df["CHR"].map(_chk_num), _s=df["CHR"].astype(str)).sort_values(["_k","_s",bp], kind="stable").drop(columns=["_k","_s"]).reset_index(drop=True)

def _norm_chr(s): return s.astype(str).str.strip().str.replace(r"^(?i:chr)", "", regex=True)

def _rt(raw):
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b": return gzip.decompress(raw).decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")

def _split_line(line):
    if "\t" in line: return [x.strip() for x in line.split("\t") if x.strip()]
    if "," in line and line.count(",") > line.count(" "): return [x.strip() for x in line.split(",") if x.strip()]
    return line.split()

def _ca(a):
    a = str(a).strip() if a is not None else "0"; return "0" if a in ("",".","-","NA","nan","None","0","00") else a

def _sample_blocks(n, max_snp, block=200, seed=42):
    if n <= max_snp: return np.arange(n)
    nb = int(np.ceil(n/block)); k = max(1, max_snp//block)
    ch = np.sort(np.random.default_rng(seed).choice(nb, size=min(k, nb), replace=False))
    return np.concatenate([np.arange(b*block, min((b+1)*block, n)) for b in ch])

# ============================================================
# PARSERS
# ============================================================
def _parse_call(cell):
    s = str(cell).strip().upper()
    if not s or s in (".","-","0","00","NA","N/A","NAN","NONE","NC","NOCALL","--","?"): return ("0","0")
    for sep in ("/","_","|",":"):
        if sep in s:
            parts = [x.strip() for x in s.split(sep)]
            if len(parts) == 2 and parts[0] and parts[1]: return (_ca(parts[0]), _ca(parts[1]))
            return ("0","0")
    try:
        v = int(float(s))
        if v == 0: return ("A","A")
        if v == 1: return ("A","B")
        if v == 2: return ("B","B")
        return ("0","0")
    except (ValueError, TypeError): pass
    if len(s) == 2 and all(c in "ACGT" for c in s): return (s[0], s[1])
    if len(s) == 1 and s in "ACGT": return (s, s)
    return ("0","0")

def parse_axiom(raw_bytes, fn=""):
    raw = gzip.decompress(raw_bytes) if (fn.endswith(".gz") or raw_bytes[:2] == b"\x1f\x8b") else raw_bytes
    text = raw.decode("utf-8", errors="replace")
    lines = [l.rstrip("\r\n") for l in text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines: raise ValueError("Fichier vide")
    def _sp(line): return line.split("\t") if "\t" in line else line.split(",") if "," in line else line.split()
    first = _sp(lines[0])
    hdr_kw = {"sample","sample_id","sampleid","sample filename","probeset_id","probe_set_id","probe set id","marker","snp","snp_id","id","axiom_id","probeset"}
    is_header = bool(first) and first[0].lower() in hdr_kw
    header = _sp(lines[0]) if is_header else None
    data_lines = lines[1:] if is_header else lines
    if len(data_lines) < 2: raise ValueError("Pas assez de lignes")
    def looks_like_call(cell):
        c = str(cell).strip().upper()
        if c in ("AA","AB","BA","BB","A","B","H","N","NC","NOCALL","N/A","0","1","2","-1",".","--","?",""): return True
        if len(c) == 2 and all(x in "ACGT" for x in c): return True
        for sep in ("/","_","|",":"):
            if sep in c and len(c) <= 5: return True
        try: int(float(c)); return True
        except (ValueError, TypeError): return False
    sample_row = _sp(data_lines[0])[1:]; n_check = min(len(sample_row), 30)
    ratio = sum(1 for c in sample_row[:n_check] if looks_like_call(c)) / max(n_check, 1)
    if ratio > 0.5:
        snp_ids, calls_by_snp = [], []
        for line in data_lines:
            p = _sp(line)
            if len(p) < 2: continue
            snp_ids.append(p[0].strip()); calls_by_snp.append(p[1:])
        if not snp_ids: raise ValueError("Aucun SNP")
        n_samp = len(calls_by_snp[0]); sample_names = header[1:] if (header and len(header) > 1) else [f"S{i+1:04d}" for i in range(n_samp)]
        if len(sample_names) < n_samp: sample_names += [f"S{i+1:04d}" for i in range(len(sample_names), n_samp)]
        sample_names = sample_names[:n_samp]
        for i in range(len(calls_by_snp)):
            if len(calls_by_snp[i]) < n_samp: calls_by_snp[i] += ["."]*(n_samp-len(calls_by_snp[i]))
            elif len(calls_by_snp[i]) > n_samp: calls_by_snp[i] = calls_by_snp[i][:n_samp]
        gtypes = [[_parse_call(calls_by_snp[j][i]) for j in range(len(snp_ids))] for i in range(n_samp)]
        return sample_names, gtypes, len(snp_ids)
    samples, gtypes = [], []
    for line in data_lines:
        p = _sp(line)
        if len(p) < 2: continue
        samples.append(p[0].strip()); gtypes.append([_parse_call(c) for c in p[1:]])
    if not samples: raise ValueError("Aucun échantillon")
    c = Counter([len(r) for r in gtypes]).most_common(1)[0][0]
    for i in range(len(gtypes)):
        if len(gtypes[i]) < c: gtypes[i] += [("0","0")]*(c-len(gtypes[i]))
        elif len(gtypes[i]) > c: gtypes[i] = gtypes[i][:c]
    return samples, gtypes, c

def parse_map(b):
    text = _rt(b); rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"): continue
        p = _split_line(line)
        if not p or p[0].lower().lstrip("#") in ("chr","chrom","chromosome","snp","marker","id"): continue
        if len(p) >= 4:
            try: rows.append({"CHR": str(p[0]), "SNP": p[1], "CM": float(p[2]) if p[2] not in ("0",".","NA","") else 0.0, "BP": int(float(p[3]))}); continue
            except (ValueError, IndexError): pass
        if len(p) == 3:
            try: rows.append({"CHR": str(p[0]), "SNP": p[1], "CM": 0.0, "BP": int(float(p[2]))}); continue
            except (ValueError, IndexError): pass
    if not rows: raise ValueError(".map vide")
    df = pd.DataFrame(rows); df["CHR"] = _norm_chr(df["CHR"])
    if df["BP"].max() < 10000: df["BP"] = df["BP"] * 1_000_000
    return df

def parse_ped(b, n_map=None):
    text = _rt(b); lines = [l for l in text.splitlines() if l.strip()]
    if not lines: raise ValueError(".ped vide")
    first = _split_line(lines[0])
    is_header = bool(first) and first[0].lower() in ("fid","sample","sample_id","sampleid","iid","animal","id","family","individual")
    if is_header: lines = lines[1:]
    first_data = _split_line(lines[0]); is_plink = False
    if len(first_data) >= 6:
        try: int(first_data[2]); int(first_data[3]); int(first_data[4]); is_plink = True
        except (ValueError, IndexError): pass
    if is_plink: n_ped = (len(first_data)-6)//2; start_idx, id_idx, fid_col = 6, 1, 0
    else: n_ped = (len(first_data)-1)//2; start_idx, id_idx, fid_col = 1, 0, None
    if n_ped < 1: raise ValueError(f"Colonnes insuffisantes : {len(first_data)}")
    n = n_ped if n_map is None else min(n_map, n_ped); ec = start_idx + 2*n
    fids, iids, allele_to_idx, gt_rows = [], [], {}, []
    for line in lines:
        p = _split_line(line)
        if len(p) < ec: p = p + ["0"]*(ec-len(p))
        fids.append(p[fid_col] if fid_col is not None else "FAM"); iids.append(p[id_idx])
        row = np.full(2*n, -1, dtype=np.int32)
        for k, a in enumerate(p[start_idx:start_idx+2*n]):
            if a not in ("0",".","NA","","N","N/N"):
                if a not in allele_to_idx: allele_to_idx[a] = len(allele_to_idx)
                row[k] = allele_to_idx[a]
        gt_rows.append(row)
    if not iids: raise ValueError("Aucun individu")
    mat = np.stack(gt_rows); a1, a2 = mat[:,0::2], mat[:,1::2]
    miss = (a1 == -1) | (a2 == -1); ok = ~miss; K = len(allele_to_idx)
    if K == 0: raise ValueError("Aucun allèle")
    cnt = np.zeros((K, n), dtype=np.int32)
    for k in range(K): cnt[k] = ((a1 == k) & ok).sum(0) + ((a2 == k) & ok).sum(0)
    idx_to_allele = {v:k for k,v in allele_to_idx.items()}
    order = np.argsort(-cnt, axis=0, kind="stable")
    major = order[0]; minor = order[1] if K > 1 else order[0]
    A1 = np.array([idx_to_allele[major[j]] if cnt[major[j],j] > 0 else "." for j in range(n)])
    A2 = np.array([idx_to_allele[minor[j]] if cnt[minor[j],j] > 0 else "." for j in range(n)])
    gt = ((a1 == minor[None,:]).astype(np.float32) + (a2 == minor[None,:]).astype(np.float32)); gt[miss] = np.nan
    return gt, pd.DataFrame({"FID": fids, "IID": iids}), A1.tolist(), A2.tolist()

def parse_vcf(b):
    samples, rows, gts_list = [], [], []
    for line in _rt(b).splitlines():
        if line.startswith("#CHROM"): samples = line.split("\t")[9:]
        elif line.strip() and not line.startswith("#"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 10 and "," not in p[4]:
                fmt = p[8].split(":")
                if "GT" not in fmt: continue
                gi = fmt.index("GT"); gts = np.full(len(samples), np.nan, dtype=np.float32)
                for k, s in enumerate(p[9:9+len(samples)]):
                    try:
                        al = s.split(":")[gi].replace("|","/").split("/")
                        if not any(a in (".","") for a in al): gts[k] = sum(int(a) for a in al)
                    except (ValueError, IndexError): pass
                try: bp = int(float(p[1]))
                except (ValueError, TypeError): continue
                rows.append({"CHR": str(p[0]), "SNP": p[2] if p[2] not in ("",".") else f"{p[0]}:{p[1]}", "CM": 0.0, "BP": bp, "A1": p[3], "A2": p[4]}); gts_list.append(gts)
    if not rows: raise ValueError("Aucun variant VCF")
    gt = np.column_stack(gts_list).astype(np.float32)
    for j in range(gt.shape[1]):
        v = gt[:,j][~np.isnan(gt[:,j])]
        if len(v) > 0 and v.mean()/2.0 > 0.5: gt[:,j] = 2.0 - gt[:,j]; rows[j]["A1"], rows[j]["A2"] = rows[j]["A2"], rows[j]["A1"]
    df_snp = pd.DataFrame(rows); df_snp["CHR"] = _norm_chr(df_snp["CHR"])
    return gt, pd.DataFrame({"FID": samples, "IID": samples}), df_snp

def _gff_attr(a, key):
    for part in a.split(";"):
        part = part.strip()
        if part.startswith(key+"="): return part[len(key)+1:]
    return None

def parse_gff3(b):
    genes = []
    for line in _rt(b).splitlines():
        if line.startswith("#") or not line.strip(): continue
        p = line.split("\t")
        if len(p) < 9 or p[2] not in ("gene","ncRNA_gene","pseudogene"): continue
        a = p[8]; name = None
        if 'gene_name "' in a: name = a.split('gene_name "')[1].split('"')[0]
        elif 'gene_id "' in a: name = a.split('gene_id "')[1].split('"')[0]
        if not name:
            for k in ("Name","gene","gene_name","ID"):
                name = _gff_attr(a, k)
                if name: break
        if not name: name = "Unknown"
        for pre in ("gene:","gene-"):
            if name.startswith(pre): name = name[len(pre):]
        chrom = p[0][3:] if p[0].lower().startswith("chr") else p[0]
        try: genes.append({"CHR": str(chrom), "START": int(p[3]), "END": int(p[4]), "GENE": name})
        except ValueError: continue
    return pd.DataFrame(genes, columns=["CHR","START","END","GENE"]).drop_duplicates().reset_index(drop=True)

def load_pheno(b, fn=""):
    raw = gzip.decompress(b) if fn.endswith(".gz") else b
    text = raw.decode("utf-8", errors="replace"); sep = "\t" if text.count("\t") > text.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python")

def match_pheno(ind_df, pheno_df, mapping_df=None):
    ind = ind_df.copy(); ind["_k"] = ind["IID"].astype(str)
    if mapping_df is not None:
        p, m = pheno_df.copy(), mapping_df.copy()
        id_c = next((c for c in ["Animal","IID","id","ID"] if c in p.columns), None)
        if id_c is None: raise ValueError("Colonne ID absente.")
        if id_c != "IID": p = p.drop(columns=["IID"], errors="ignore")
        p[id_c] = p[id_c].astype(str)
        map_a = next((c for c in m.columns if c != "IID"), None)
        if map_a is None or "IID" not in m.columns: raise ValueError("Mapping invalide")
        m[map_a] = m[map_a].astype(str); m["IID"] = m["IID"].astype(str)
        pm = p.merge(m, left_on=id_c, right_on=map_a, how="left"); pm["_k"] = pm["IID"].astype(str); pm = pm.drop_duplicates("_k")
        return ind[["_k","FID","IID"]].merge(pm.drop(columns=["IID","FID"], errors="ignore"), on="_k", how="left").drop(columns="_k"), id_c
    id_c = next((c for c in ["IID","id","ID","sample","Animal","animal"] if c in pheno_df.columns), None)
    if id_c is None: raise ValueError("Aucune colonne ID.")
    p = pheno_df.copy(); p["_k"] = p[id_c].astype(str); p = p.drop_duplicates("_k")
    return ind[["_k","FID","IID"]].merge(p.drop(columns=[id_c,"FID"], errors="ignore"), on="_k", how="left").drop(columns="_k"), id_c

def gen_demo(n_ind=150, n_snp=2000, n_pop=4, seed=42, autosomes=None, ars=None):
    rng = np.random.default_rng(seed)
    if not autosomes or not ars: autosomes = [str(i) for i in range(1,30)]; ars = {c:100.0 for c in autosomes}
    pops = [f"POP_{i}" for i in range(1, min(n_pop,10)+1)]
    pa = rng.beta(0.6, 0.6, n_snp); gt = np.zeros((n_ind, n_snp), dtype=np.float32); ir = []; k = 0
    for pi_, p in enumerate(pops):
        pk = np.clip(pa + rng.normal(0, 0.15, n_snp), 0.02, 0.98)
        for i in range(n_ind // len(pops) + (1 if pi_ < n_ind % len(pops) else 0)):
            gt[k] = (rng.random(n_snp) < pk).astype(np.int8) + (rng.random(n_snp) < pk).astype(np.int8)
            ir.append({"FID": p, "IID": f"{p}_{i+1:03d}"}); k += 1
    gt[rng.random(gt.shape) < 0.02] = np.nan
    ch = rng.choice(autosomes, n_snp, p=_autw(ars, autosomes))
    bp = np.array([rng.integers(1, int(ars[c]*1_000_000)) for c in ch], dtype=np.int64)
    snp = pd.DataFrame({"CHR": ch.astype(str), "SNP": [f"rs{i:07d}" for i in range(n_snp)], "CM": bp/1e6, "BP": bp, "A1": "A", "A2": "G"})
    order = np.lexsort((bp, snp["CHR"].map(_chk_num).values))
    return gt[:,order], pd.DataFrame(ir), snp.iloc[order].reset_index(drop=True)

# ============================================================
# QC
# ============================================================
def miss_ind(g): return np.isnan(g).mean(axis=1)
def miss_snp(g): return np.isnan(g).mean(axis=0)
def maf(g): p = np.nanmean(g, 0)/2.0; return np.minimum(p, 1.0-p)
def het(g):
    ok = ~np.isnan(g); n = ok.sum(1); return np.where(n > 0, ((g == 1) & ok).sum(1)/np.maximum(n,1), np.nan)

def hwe_exact(nh, n0, n2):
    n = nh + n0 + n2
    if n == 0 or (nh == 0 and (n0 == 0 or n2 == 0)): return 1.0
    rare = 2*min(n0,n2) + nh; mid = (rare*(2*n-rare))//(2*n)
    if mid % 2 != rare % 2: mid += 1
    pr = np.zeros(rare+1); pr[mid] = 1.0; s = 1.0
    ch, cr, cc = mid, (rare-mid)//2, n-mid-(rare-mid)//2
    while ch <= rare-2: pr[ch+2] = pr[ch]*4*cr*cc/((ch+2)*(ch+1)); s += pr[ch+2]; ch += 2; cr -= 1; cc -= 1
    ch, cr, cc = mid, (rare-mid)//2, n-mid-(rare-mid)//2
    while ch >= 2: pr[ch-2] = pr[ch]*ch*(ch-1)/(4*(cr+1)*(cc+1)); s += pr[ch-2]; ch -= 2; cr += 1; cc += 1
    po = pr[nh] if nh < len(pr) else 0.0
    return float(min(pr[pr <= po+1e-7].sum()/s, 1.0))

@cache_data
def hwe_pvals(g):
    pv = np.full(g.shape[1], np.nan)
    n0 = ((g == 0) & ~np.isnan(g)).sum(0); n1 = ((g == 1) & ~np.isnan(g)).sum(0); n2 = ((g == 2) & ~np.isnan(g)).sum(0); nv = n0+n1+n2
    if g.shape[1] <= 20000:
        for j in range(g.shape[1]):
            if nv[j] >= 5: pv[j] = hwe_exact(int(n1[j]), int(n0[j]), int(n2[j]))
    else:
        p = (n1+2*n2)/(2.0*np.where(nv>0, nv, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            eh = 2.0*p*(1.0-p)*nv; ok = (eh >= 5) & np.isfinite(eh); chi2 = (np.abs(n1-eh)-0.5)**2/eh; pv[ok] = stats.chi2.sf(chi2[ok], df=1)
    return pv

@cache_data
def hwe_by_pop(gt, labels, min_n=10):
    labels = np.asarray(labels).astype(str); cols = {}
    for p in sorted(np.unique(labels)):
        m = labels == p
        if m.sum() < min_n: continue
        g = gt[m]; ok = ~np.isnan(g)
        n0 = ((g == 0) & ok).sum(0); n1 = ((g == 1) & ok).sum(0); n2 = ((g == 2) & ok).sum(0); pv = np.full(g.shape[1], np.nan)
        for j in range(g.shape[1]):
            if n0[j]+n1[j]+n2[j] >= 5: pv[j] = hwe_exact(int(n1[j]), int(n0[j]), int(n2[j]))
        cols[p] = pv
    return pd.DataFrame(cols)

def sync_dims(gt, ind_df, snp_df):
    n = min(gt.shape[0], len(ind_df)); m = min(gt.shape[1], len(snp_df))
    return gt[:n,:m], ind_df.iloc[:n].reset_index(drop=True), snp_df.iloc[:m].reset_index(drop=True)

def apply_qc(gt, ind_df, snp_df, params, autosomes, keep_autosomes=True, rem_out=True):
    gt, ind_df, snp_df = sync_dims(gt, ind_df, snp_df); n0, m0 = gt.shape; tr = {}
    if keep_autosomes:
        auto = {str(a).upper() for a in autosomes}
        mask = snp_df["CHR"].astype(str).str.upper().str.replace(r"^CHR","", regex=True).isin(auto).values
        gt = gt[:,mask]; snp_df = snp_df[mask].reset_index(drop=True); tr["autosomes"] = int(m0-gt.shape[1])
    m = gt.shape[1]
    k = miss_snp(gt) <= params["geno"]; gt = gt[:,k]; snp_df = snp_df[k].reset_index(drop=True); tr["geno"] = int(m-gt.shape[1]); m = gt.shape[1]
    k = miss_ind(gt) <= params["mind"]; gt = gt[k]; ind_df = ind_df[k].reset_index(drop=True); tr["mind"] = int(n0-gt.shape[0])
    mf = maf(gt); k = np.isfinite(mf) & (mf >= params["maf"]); gt = gt[:,k]; snp_df = snp_df[k].reset_index(drop=True); tr["maf"] = int(m-gt.shape[1]); m = gt.shape[1]
    pv = hwe_pvals(gt); k = np.isnan(pv) | (pv >= params["hwe"])
    gt = gt[:,k]; snp_df = snp_df[k].reset_index(drop=True); tr["hwe"] = int(m-gt.shape[1])
    n_pre = int(gt.shape[0])
    if rem_out:
        h = het(gt); kp = np.ones(len(h), dtype=bool)
        for f in ind_df["FID"].unique():
            msk = (ind_df["FID"] == f).values; sub = h[msk]
            if len(sub) >= 5 and np.nanstd(sub) > 1e-9:
                z = (sub-np.nanmean(sub))/np.nanstd(sub); idx = np.where(msk)[0]; kp[idx[np.abs(z) > params["het_sd"]]] = False
        gt = gt[kp]; ind_df = ind_df[kp].reset_index(drop=True); tr["het"] = int(n_pre-len(ind_df))
    gc.collect()
    return gt, ind_df, snp_df, {"n_ind_final": int(gt.shape[0]), "n_snp_final": int(gt.shape[1]), "trace": tr}

# ============================================================
# LD PRUNING
# ============================================================
@cache_data
def ld_pruning(gt, chrs, w=50, step=5, r2=0.2):
    n = gt.shape[1]; keep = np.ones(n, dtype=bool)
    if n < 3: return keep
    chrs = np.asarray(chrs).astype(str); X = _std_geno(gt).astype(np.float32); N = X.shape[0]
    for c in pd.unique(chrs):
        idx = np.where(chrs == c)[0]
        for s in range(0, len(idx), step):
            win = idx[s:s+w]; win = win[keep[win]]
            if len(win) < 2: continue
            R2 = ((X[:,win].T @ X[:,win])/N)**2
            for a in range(len(win)-1):
                if not keep[win[a]]: continue
                hits = np.where(R2[a, a+1:] > r2)[0] + a + 1; keep[win[hits]] = False
    return keep

@cache_data
def ld_prune_mask(gt, snp_df, autosomes, r2=0.2, w=50, step=5):
    chrs = snp_df["CHR"].astype(str).str.upper().str.replace(r"^CHR","", regex=True).values
    auto = {str(a).upper() for a in autosomes}
    on_auto = np.isin(chrs, list(auto))
    keep = np.zeros(len(chrs), dtype=bool)
    idx_auto = np.where(on_auto)[0]
    if len(idx_auto) < 10: keep[idx_auto] = True; return keep
    keep[idx_auto] = ld_pruning(gt[:,idx_auto], chrs[idx_auto], w=w, step=step, r2=r2)
    return keep

def view_ld_pruned(gt, snp_df, autosomes, **kw):
    m = ld_prune_mask(gt, snp_df, autosomes, **kw)
    return gt[:,m], snp_df[m].reset_index(drop=True)

def caption_ld(gt_in, gt_out, r2):
    pct = 100*(1-gt_out.shape[1]/max(gt_in.shape[1],1))
    st.caption(f"🧬 **LD pruning auto** (r²>{r2}) : {gt_in.shape[1]:,} → {gt_out.shape[1]:,} SNPs ({pct:.1f}% retirés)")

# ============================================================
# DIVERSITÉ & STRUCTURE
# ============================================================
def _wc_abc(gt, labels, min_n=2):
    labels = np.asarray(labels).astype(str); pops = np.unique(labels); m = gt.shape[1]
    n = np.zeros((len(pops), m)); p = np.zeros_like(n); h = np.zeros_like(n)
    for k, pop in enumerate(pops):
        g = gt[labels == pop]; ok = ~np.isnan(g); nn = ok.sum(0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            pk = np.where(nn>0, np.nansum(g,0)/(2*nn), 0.0); hk = np.where(nn>0, ((g == 1) & ok).sum(0)/nn, 0.0)
        use = nn >= min_n
        n[k] = np.where(use, nn, 0.0); p[k] = np.where(use, pk, 0.0); h[k] = np.where(use, hk, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (n>0).sum(0).astype(float); nsum = n.sum(0); nbar = nsum/r; pbar = (n*p).sum(0)/nsum
        s2 = (n*(p-pbar)**2).sum(0)/(nbar*(r-1)); hbar = (n*h).sum(0)/nsum
        nc = (r*nbar-(n**2).sum(0)/(r*nbar))/(r-1)
        a = (nbar/nc)*(s2-(1.0/(nbar-1))*(pbar*(1-pbar)-(r-1)/r*s2-hbar/4))
        b = (nbar/(nbar-1))*(pbar*(1-pbar)-(r-1)/r*s2-(2*nbar-1)/(4*nbar)*hbar)
        den = a+b+hbar/2.0
    valid = (r >= 2) & (nbar > 1) & (nc > 0) & np.isfinite(den) & np.isfinite(a)
    return np.where(valid, a, np.nan), np.where(valid, b, np.nan), np.where(valid, hbar/2.0, np.nan)

def _wc_components(gt, labels, min_n=2):
    a, b, c = _wc_abc(gt, labels, min_n); return a, a+b+c

@cache_data
def fst_snp_wc(gt, labels):
    a, den = _wc_components(gt, labels)
    with np.errstate(invalid="ignore", divide="ignore"): f = a/den
    return np.where(np.isfinite(f) & (den > 1e-9), np.clip(f, 0.0, None), np.nan)

def fst_global_wc(gt, labels):
    a, den = _wc_components(gt, labels); ok = np.isfinite(a) & np.isfinite(den)
    return float(max(0.0, a[ok].sum()/den[ok].sum())) if ok.any() and den[ok].sum() > 0 else np.nan

@cache_data
def pca_analysis(gt, nc=10):
    X = impute_mean(gt); X = X-X.mean(0); n = min(nc, X.shape[0]-1, X.shape[1]); p = PCA(n_components=n)
    return p.fit_transform(X), p.explained_variance_ratio_*100.0

@cache_data
def mds_analysis(gt, nc=5):
    X = impute_mean(gt); n = X.shape[0]; D = np.zeros((n,n), dtype=np.float32)
    for i in range(n): D[i] = np.abs(X-X[i]).sum(1)/X.shape[1]
    kw = dict(n_components=nc, random_state=42, n_init=1, max_iter=300)
    try: return SklearnMDS(dissimilarity="precomputed", normalized_stress=False, **kw).fit_transform(D)
    except Exception: return SklearnMDS(metric="precomputed", **kw).fit_transform(D)

@cache_data
def compute_diversity(gt, labels):
    labels = np.asarray(labels).astype(str); rows = []
    for p in np.unique(labels):
        sub = gt[labels == p]
        if sub.shape[0] < 3: continue
        ok = ~np.isnan(sub); n = ok.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            freq = np.where(n>0, np.nansum(sub,0)/(2*n), np.nan); ho_snp = np.where(n>0, ((sub == 1) & ok).sum(0)/n, np.nan)
        v = np.isfinite(freq)
        if not v.any(): continue
        f = freq[v]; q = 1-f; he = float(np.mean(2*f*q)); ho = float(np.nanmean(ho_snp[v]))
        pic = float(np.mean(1-(f**2+q**2)-2*(f**2)*(q**2)))
        rows.append({"Population": p, "N": int(sub.shape[0]), "Ho": round(ho,4), "He": round(he,4), "Fis": round(1-ho/he,4) if he > 0 else 0.0, "PIC": round(pic,4), "MAF": round(float(np.mean(np.minimum(f,q))),4), "Polymorphes": int(((f>0)&(f<1)).sum())})
    return pd.DataFrame(rows)

@cache_data
def king_mat(gt):
    ok = ~np.isnan(gt); H = ((gt==1)&ok).astype(np.float32); G0 = ((gt==0)&ok).astype(np.float32); G2 = ((gt==2)&ok).astype(np.float32)
    h = H.sum(1).astype(np.float64); BH = H@H.T; ibs0 = G0@G2.T + G2@G0.T; d = h[:,None] + h[None,:]
    with np.errstate(divide="ignore", invalid="ignore"): K = (BH-2*ibs0)/d
    K[~np.isfinite(K)] = 0.0; np.fill_diagonal(K, 0.5); return K

@cache_data
def grm_vanraden(gt):
    X = impute_mean(gt); p = X.mean(0)/2.0; Z = X-2*p; den = 2*np.sum(p*(1-p))
    if den <= 0: return np.eye(gt.shape[0])
    return ((Z@Z.T)/den).astype(np.float64)

@cache_data
def admixture_nmf(gt, K=3, seed=42):
    m = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=500)
    W = m.fit_transform(np.clip(impute_mean(gt), 0.0, 2.0)); s = W.sum(1, keepdims=True); s[s == 0] = 1.0
    return W/s, m.components_

@cache_data
def choose_best_k(gt, k_min=2, k_max=8, seed=42):
    X = np.clip(impute_mean(gt), 0.0, 2.0); rows = []
    for k in range(k_min, k_max+1):
        m = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=400); W = m.fit_transform(X)
        rows.append({"K": k, "RMSE": float(np.sqrt(np.mean((X-W@m.components_)**2)))})
    return pd.DataFrame(rows)

def _pair_r2(X, bp, chrs, win=100, max_bp=None):
    n = X.shape[1]; N = X.shape[0]; D, R = [], []
    for i in range(n-1):
        j1 = min(i+1+win, n); same = chrs[i+1:j1] == chrs[i]
        if not same.any(): continue
        d = np.abs(bp[i+1:j1]-bp[i]); ok = same & (d > 0)
        if max_bp is not None: ok &= (d <= max_bp)
        if ok.any():
            r = (X[:,i+1:j1][:,ok].T @ X[:,i])/N; D.append(d[ok]); R.append(r.astype(np.float64)**2)
    return (np.concatenate(D), np.concatenate(R)) if D else (np.array([]), np.array([]))

def _sorted_snps(gt, snp_df):
    s = snp_df.reset_index(drop=True).copy(); s["CHR"] = s["CHR"].astype(str); s["BP"] = s["BP"].astype(float)
    order = np.lexsort((s["BP"].values, s["CHR"].map(_chk_num).values))
    return gt[:,order], s.iloc[order].reset_index(drop=True)

@cache_data
def ld_decay(gt, snp_df, max_kb=1000, max_snp=20000):
    if gt.shape[1] < 10: return pd.DataFrame(columns=["dist_kb","r2","count"])
    gt, s = _sorted_snps(gt, snp_df)
    idx = _sample_blocks(gt.shape[1], max_snp); gt = gt[:,idx]; s = s.iloc[idx].reset_index(drop=True)
    X = _std_geno(gt); dist, r2 = _pair_r2(X, s["BP"].values, s["CHR"].values, 100, max_kb*1000)
    if len(dist) == 0: return pd.DataFrame(columns=["dist_kb","r2","count"])
    df = pd.DataFrame({"dist_kb": dist/1000.0, "r2": r2})
    df["bin"] = pd.cut(df["dist_kb"], bins=np.linspace(0, max_kb, 30), labels=False, include_lowest=True)
    return df.groupby("bin").agg(dist_kb=("dist_kb","mean"), r2=("r2","mean"), count=("r2","count")).dropna().sort_values("dist_kb").reset_index(drop=True)

@cache_data
def ne_historical(gt, snp_df, max_snp=10000):
    N = gt.shape[0]
    if N < 5 or gt.shape[1] < 20: return pd.DataFrame({"GenAgo": [], "Ne": []})
    gt, s = _sorted_snps(gt, snp_df); idx = _sample_blocks(gt.shape[1], max_snp); gt = gt[:,idx]; s = s.iloc[idx].reset_index(drop=True)
    X = _std_geno(gt); d, r2 = _pair_r2(X, s["BP"].values, s["CHR"].values, 50, 50e6); d = d/1e6
    m = (d >= 0.001) & (d <= 50)
    if m.sum() < 20: return pd.DataFrame({"GenAgo": [], "Ne": []})
    d, r2 = d[m], r2[m]; bins = np.logspace(np.log10(d.min()), np.log10(d.max()), 20); rows = []
    for k in range(len(bins)-1):
        sel = (d >= bins[k]) & (d < bins[k+1])
        if sel.sum() >= 10: rows.append((np.sqrt(bins[k]*bins[k+1]), r2[sel].mean()))
    if len(rows) < 3: return pd.DataFrame({"GenAgo": [], "Ne": []})
    dm, rm = np.array(rows).T; c = dm*0.01; r2a = np.clip(rm-1.0/N, 1e-6, 0.999)
    ne = (1.0/r2a-1.0)/(4.0*c); t = 1.0/(2.0*c)
    df = pd.DataFrame({"GenAgo": t, "Ne": ne}).replace([np.inf,-np.inf], np.nan).dropna(); df = df[df["Ne"] > 0].sort_values("GenAgo").reset_index(drop=True)
    if len(df) > 3: df["Ne"] = df["Ne"].rolling(3, center=True, min_periods=1).mean()
    return df

@cache_data
def detect_roh(gt, chrs, bps, min_snps=30, min_kb=500.0):
    chrs = np.asarray(chrs).astype(str); bps = np.asarray(bps).astype(np.int64); n_ind = gt.shape[0]; rohs = []; fb = np.zeros(n_ind); tot = 0.0
    for c in pd.unique(chrs):
        idxs = np.where(chrs == c)[0]
        if len(idxs) < min_snps: continue
        idxs = idxs[np.argsort(bps[idxs], kind="stable")]; pos = bps[idxs]; sp = pos[-1]-pos[0]
        if sp < 1: continue
        tot += sp; G = gt[:,idxs]
        for i in range(n_ind):
            bnd = np.concatenate(([-1], np.where(G[i] == 1)[0], [len(idxs)]))
            for a, b in zip(bnd[:-1], bnd[1:]):
                s_, e_ = a+1, b-1
                if e_ < s_: continue
                nr = e_-s_+1; lkb = (pos[e_]-pos[s_])/1000.0
                if nr >= min_snps and lkb >= min_kb:
                    rohs.append({"IID_idx": i, "CHR": c, "start_bp": int(pos[s_]), "end_bp": int(pos[e_]), "n_snp": int(nr), "length_kb": float(lkb)}); fb[i] += pos[e_]-pos[s_]
    cols = ["IID_idx","CHR","start_bp","end_bp","n_snp","length_kb"]
    return pd.DataFrame(rohs, columns=cols), (fb/tot if tot > 0 else fb)

@cache_data
def fst_pairwise(gt, labels, use_wc84=True):
    labels = np.asarray(labels).astype(str); pops = sorted(np.unique(labels)); K = len(pops); M = np.full((K,K), np.nan); np.fill_diagonal(M, 0.0)
    for i in range(K):
        for j in range(i+1, K):
            mi, mj = labels == pops[i], labels == pops[j]
            if use_wc84: v = fst_global_wc(gt[mi|mj], labels[mi|mj])
            else:
                g1, g2 = gt[mi], gt[mj]; p1 = np.nanmean(g1,0)/2.0; p2 = np.nanmean(g2,0)/2.0; n1 = (~np.isnan(g1)).sum(0); n2 = (~np.isnan(g2)).sum(0)
                dn = np.where(n1+n2>0, n1+n2, np.nan); pb = (p1*n1+p2*n2)/dn; hs = (2*p1*(1-p1)*n1+2*p2*(1-p2)*n2)/dn; ht = 2*pb*(1-pb)
                ok = np.isfinite(ht) & np.isfinite(hs); v = float(max(0.0, (ht[ok]-hs[ok]).sum()/ht[ok].sum())) if ok.any() and ht[ok].sum() > 0 else np.nan
            M[i,j] = M[j,i] = v
    return M, pops

@cache_data
def windowed_fst(gt, snp_df, labels, w=50, step=25):
    a, den = _wc_components(gt, labels); snp = snp_df.reset_index(drop=True).copy(); n = min(len(snp), len(a)); snp = snp.iloc[:n].copy(); a = a[:n]; den = den[:n]
    snp["_a"] = a; snp["_d"] = den
    with np.errstate(invalid="ignore", divide="ignore"): snp["FST"] = np.where(np.isfinite(den) & (den > 1e-9), np.clip(a/den, 0.0, None), np.nan)
    res = []
    for c in snp["CHR"].astype(str).unique():
        sub = snp[snp["CHR"].astype(str) == c].sort_values("BP").reset_index(drop=True)
        if len(sub) < w: continue
        A, Dn, F, B, S = sub["_a"].values, sub["_d"].values, sub["FST"].values, sub["BP"].values, sub["SNP"].values
        for i in range(0, len(sub)-w+1, step):
            sl = slice(i, i+w); okm = np.isfinite(A[sl]) & np.isfinite(Dn[sl])
            if okm.sum() < 5: continue
            dd = Dn[sl][okm].sum()
            if dd <= 1e-9: continue
            f_sl = F[sl]; top = int(np.nanargmax(f_sl)) if np.isfinite(f_sl).any() else 0
            res.append({"CHR": c, "START": int(B[sl].min()), "END": int(B[sl].max()), "MID": int((B[sl].min()+B[sl].max())/2), "N_SNP": w, "MEAN_FST": float(max(0.0, A[sl][okm].sum()/dd)), "MAX_FST": float(np.nanmax(f_sl)) if np.isfinite(f_sl).any() else np.nan, "TOP_SNP": S[sl][top]})
    return pd.DataFrame(res, columns=["CHR","START","END","MID","N_SNP","MEAN_FST","MAX_FST","TOP_SNP"])

@cache_data
def detect_outliers(win_df, z_thr=3.0):
    df = win_df.copy()
    if df.empty: df["Z_FST"] = []; df["OUTLIER"] = []; return df
    mu, sd = df["MEAN_FST"].mean(), df["MEAN_FST"].std()
    df["Z_FST"] = (df["MEAN_FST"]-mu)/sd if sd > 1e-9 else 0.0
    df["OUTLIER"] = df["Z_FST"] >= z_thr
    return df.sort_values("MEAN_FST", ascending=False).reset_index(drop=True)

@cache_data
def tajimas_window(genos):
    ni, ns = genos.shape
    if ns < 3 or ni < 4: return np.nan, 0, np.nan, np.nan
    S = 0; piv = []; thv = []
    for j in range(ns):
        col = genos[:,j]; col = col[~np.isnan(col)]
        if len(col) < 2: continue
        p = col.mean()/2.0; N = 2*len(col)
        if 0 < p < 1: S += 1; piv.append(2*p*(1-p)*N/(N-1)); thv.append(1.0/np.sum(1.0/np.arange(1, N)))
    if S == 0 or not piv: return np.nan, int(S), np.nan, np.nan
    pi = float(np.sum(piv)); th = float(np.sum(thv)); n = ni*2
    a1 = np.sum(1.0/np.arange(1,n)); a2 = np.sum(1.0/(np.arange(1,n)**2))
    b1 = (n+1)/(3*(n-1)); b2 = 2*(n**2+n+3)/(9*n*(n-1))
    c1 = b1-1/a1; c2 = b2-(n+2)/(a1*n)+a2/(a1**2); e1 = c1/a1; e2 = c2/(a1**2+a2)
    den = np.sqrt(e1*S+e2*S*(S-1))
    if den <= 1e-12: return np.nan, int(S), pi, th
    return float((pi-S/a1)/den), int(S), pi, th

@cache_data
def tajimas_scan(gt, snp_df, w=25, step=10):
    snp = snp_df.copy().reset_index(drop=True); n = min(gt.shape[1], len(snp)); snp = snp.iloc[:n].reset_index(drop=True); gt = gt[:,:n]; res = []
    for c in snp["CHR"].unique():
        idx = np.where(snp["CHR"].values == c)[0]
        if len(idx) < w: continue
        idx = idx[np.argsort(snp["BP"].values[idx])]
        for i in range(0, len(idx)-w+1, step):
            wi = idx[i:i+w]; D, S, pis, ths = tajimas_window(gt[:,wi])
            bpw = snp["BP"].values[wi]; span_kb = max((bpw.max()-bpw.min())/1000.0, 1e-9)
            if not np.isnan(D):
                res.append({"CHR": str(c), "START": int(bpw.min()), "END": int(bpw.max()), "MID": int((bpw.min()+bpw.max())/2), "N_SNP": len(wi), "S": S, "TAJIMA_D": D, "PI_PAR_SNP": pis/len(wi), "THETA_W_PAR_SNP": ths/len(wi), "PI_PAR_KB": pis/span_kb})
    return pd.DataFrame(res, columns=["CHR","START","END","MID","N_SNP","S","TAJIMA_D","PI_PAR_SNP","THETA_W_PAR_SNP","PI_PAR_KB"])

@cache_data
def tajimas_scan_phys(gt, snp_df, window_kb=500, step_kb=250, min_snps=10):
    snp = snp_df.copy().reset_index(drop=True); n = min(gt.shape[1], len(snp)); snp = snp.iloc[:n].reset_index(drop=True); gt = gt[:,:n]; res = []
    step_bp = int(step_kb*1000); win_bp = int(window_kb*1000)
    for c in snp["CHR"].unique():
        idx_c = np.where(snp["CHR"].values == c)[0]
        if len(idx_c) < min_snps: continue
        idx_c = idx_c[np.argsort(snp["BP"].values[idx_c])]; bp_c = snp["BP"].values[idx_c]
        for start in range(int(bp_c[0]), int(bp_c[-1])+1, step_bp):
            end = start+win_bp; mask = (bp_c >= start) & (bp_c < end)
            if mask.sum() < min_snps: continue
            wi = idx_c[mask]; D, S, pis, ths = tajimas_window(gt[:,wi])
            bpw = bp_c[mask]; span_kb = max((bpw.max()-bpw.min())/1000.0, 1e-9)
            if not np.isnan(D):
                res.append({"CHR": str(c), "START": int(start), "END": int(end), "MID": int((start+end)//2), "N_SNP": int(len(wi)), "S": S, "TAJIMA_D": D, "PI_PAR_SNP": pis/len(wi), "THETA_W_PAR_SNP": ths/len(wi), "PI_PAR_KB": pis/span_kb})
    return pd.DataFrame(res, columns=["CHR","START","END","MID","N_SNP","S","TAJIMA_D","PI_PAR_SNP","THETA_W_PAR_SNP","PI_PAR_KB"])

def shannon_window(genos):
    ni, ns = genos.shape
    if ns == 0 or ni < 2: return np.nan
    h_vals = []
    for j in range(ns):
        col = genos[:,j][~np.isnan(genos[:,j])]
        if len(col) < 2: continue
        p = col.mean()/2.0
        if 0 < p < 1:
            q = 1.0-p; h_vals.append(-(p*np.log(p)+q*np.log(q)))
    return float(np.mean(h_vals)) if h_vals else np.nan

@cache_data
def shannon_scan_phys(gt, snp_df, window_kb=500, step_kb=250, min_snps=10):
    snp = snp_df.copy().reset_index(drop=True); n = min(gt.shape[1], len(snp)); snp = snp.iloc[:n].reset_index(drop=True); gt = gt[:,:n]; res = []
    step_bp = int(step_kb*1000); win_bp = int(window_kb*1000)
    for c in snp["CHR"].unique():
        idx_c = np.where(snp["CHR"].values == c)[0]
        if len(idx_c) < min_snps: continue
        idx_c = idx_c[np.argsort(snp["BP"].values[idx_c])]; bp_c = snp["BP"].values[idx_c]
        for start in range(int(bp_c[0]), int(bp_c[-1])+1, step_bp):
            end = start+win_bp; mask = (bp_c >= start) & (bp_c < end)
            if mask.sum() < min_snps: continue
            wi = idx_c[mask]; H = shannon_window(gt[:,wi])
            if not np.isnan(H):
                res.append({"CHR": str(c), "START": int(start), "END": int(end), "MID": int((start+end)//2), "N_SNP": int(len(wi)), "SHANNON_H": float(H)})
    return pd.DataFrame(res, columns=["CHR","START","END","MID","N_SNP","SHANNON_H"])

@cache_data
def composite_score(gt, snp_df, labels):
    fst = fst_snp_wc(gt, labels); ok = ~np.isnan(gt); n = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        hom = np.where(n>0, (((gt == 0)|(gt == 2))&ok).sum(0)/n, np.nan); mv = np.nanmean(gt, axis=0)/2.0
    mv = np.minimum(mv, 1-mv); df = snp_df.copy().reset_index(drop=True); m = min(len(df), len(fst)); df = df.iloc[:m].reset_index(drop=True)
    df["FST"] = fst[:m]; df["HOM"] = hom[:m]; df["MAF"] = mv[:m]
    def norm(x): mu, sd = np.nanmean(x), np.nanstd(x); return (x-mu)/sd if sd > 1e-9 else np.zeros_like(x)
    df["Z_FST"] = norm(df["FST"].values); df["Z_HOM"] = norm(df["HOM"].values); df["CSS"] = np.abs(df["Z_FST"]*df["Z_HOM"])
    return df.sort_values("CSS", ascending=False).reset_index(drop=True)

def _reynolds_D(g, labels, pops):
    ph = {p: np.nanmean(g[labels == p], 0)/2.0 for p in pops}; K = len(pops); D = np.zeros((K,K))
    for i in range(K):
        for j in range(i+1, K):
            pi_, pj = ph[pops[i]], ph[pops[j]]; ok = np.isfinite(pi_) & np.isfinite(pj)
            num = np.sum((pi_[ok]-pj[ok])**2); dn = np.sum(pi_[ok]*(1-pj[ok])+pj[ok]*(1-pi_[ok]))
            if dn > 0: D[i,j] = D[j,i] = -np.log(1.0-min(num/dn, 0.999999))
    return D

@cache_data
def reynolds(gt, labels):
    labels = np.asarray(labels).astype(str); pops = sorted(np.unique(labels)); return _reynolds_D(gt, labels, pops), pops

@cache_data
def upgma_bootstrap(gt, snp_df, labels, n_boot=100, seed=42):
    labels = np.asarray(labels).astype(str); pops = sorted(np.unique(labels)); K = len(pops); n_snp = gt.shape[1]
    if K < 2: raise ValueError("Au moins 2 populations requises")
    Do = _reynolds_D(gt, labels, pops); Zo = linkage(squareform(Do, checks=False), method="average"); nl = K
    def clusters(Z):
        cl = {}
        for k, row in enumerate(Z):
            i, j = int(row[0]), int(row[1])
            l = frozenset([i]) if i < nl else cl[i]; r = frozenset([j]) if j < nl else cl[j]
            cl[nl+k] = frozenset(l|r)
        return cl
    co = clusters(Zo); sc = np.zeros(len(Zo)); rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.choice(n_snp, n_snp, replace=True)
        Zb = linkage(squareform(_reynolds_D(gt[:,idx], labels, pops), checks=False), method="average")
        cb = set(clusters(Zb).values())
        for k, cl in co.items():
            if cl in cb: sc[k-nl] += 1
    supp = (sc/n_boot*100).round(0); tr = to_tree(Zo)
    def nw(node):
        if node.is_leaf(): return f"{pops[node.id]}"
        l = nw(node.get_left()); r = nw(node.get_right())
        return f"({l},{r}){int(supp[node.id-nl])}:{node.dist:.4f}" if node.id >= nl else f"({l},{r}):{node.dist:.4f}"
    return {"tree": tr, "Z": Zo, "newick": nw(tr)+";", "support": supp, "D": Do, "pops": pops}

@cache_data
def midpoint_info(D, labels):
    D = np.asarray(D, dtype=float); n = len(labels); D = (D+D.T)/2; np.fill_diagonal(D, 0); im, jm, md = 0, 1, 0.0
    for i in range(n):
        for j in range(i+1, n):
            if D[i,j] > md: md = D[i,j]; im, jm = i, j
    return {"leaf_i": labels[im], "leaf_j": labels[jm], "max_dist": float(md), "midpoint": float(md/2)}

# ============================================================
# GWAS
# ============================================================
def _logistic_irls(X, y, max_iter=25, tol=1e-6):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X@beta, -30, 30); mu = 1.0/(1.0+np.exp(-eta)); W = mu*(1-mu)+1e-9
        H = (X.T*W)@X + 1e-8*np.eye(X.shape[1])
        try: step = np.linalg.solve(H, X.T@(y-mu))
        except np.linalg.LinAlgError: return float(beta[-1]), None
        beta_new = beta+step
        if np.max(np.abs(beta_new-beta)) < tol: beta = beta_new; break
        beta = beta_new
    eta = np.clip(X@beta, -30, 30); mu = 1.0/(1.0+np.exp(-eta)); W = mu*(1-mu)+1e-9
    try:
        cov = np.linalg.inv((X.T*W)@X+1e-8*np.eye(X.shape[1])); se = float(np.sqrt(cov[-1,-1]))
    except np.linalg.LinAlgError: return float(beta[-1]), None
    return float(beta[-1]), se

def _pairwise_core(gt, snp_df, groups, ga, gb, maf_min=0.05, method="allelic", covs=None):
    groups = np.asarray(groups).astype(str); mask = np.isin(groups, [ga, gb])
    if mask.sum() < 20: raise ValueError(f"<20 individus pour {ga} vs {gb}")
    g = gt[mask]; y = (groups[mask] == ga).astype(int)
    if (y == 1).sum() < 5 or (y == 0).sum() < 5: raise ValueError("Moins de 5 individus par classe")
    snp = snp_df.reset_index(drop=True); n = min(g.shape[1], len(snp)); g = g[:,:n]; snp = snp.iloc[:n].reset_index(drop=True)
    cv = covs[mask] if covs is not None else None
    gA, gB = g[y == 1], g[y == 0]
    nA = (~np.isnan(gA)).sum(0).astype(float); nB = (~np.isnan(gB)).sum(0).astype(float)
    sA = np.nansum(gA, 0); sB = np.nansum(gB, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pA = sA/(2*nA); pB = sB/(2*nB); tot = (sA+sB)/(2*(nA+nB)); mj = np.minimum(tot, 1-tot)
        hs = (2*pA*(1-pA)*nA+2*pB*(1-pB)*nB)/(nA+nB); pb = (pA*nA+pB*nB)/(nA+nB); ht = 2*pb*(1-pb)
        fv = np.where(ht > 1e-9, np.clip((ht-hs)/ht, 0, None), 0.0)
    valid = (nA >= 5) & (nB >= 5) & np.isfinite(mj) & (mj >= maf_min)
    if method == "allelic":
        a1, a0, b1, b0 = sA, 2*nA-sA, sB, 2*nB-sB; N = a1+a0+b1+b0
        with np.errstate(invalid="ignore", divide="ignore"):
            den = (a1+a0)*(b1+b0)*(a1+b1)*(a0+b0); chi2 = N*(a1*b0-a0*b1)**2/den
        valid &= np.isfinite(chi2) & (den > 0)
        pv = np.where(valid, stats.chi2.sf(np.where(valid, chi2, 0), 1), np.nan)
    else:
        chi2 = np.full(n, np.nan); pv = np.full(n, np.nan)
        for j in np.where(valid)[0]:
            X = np.column_stack([np.ones(len(y))]+([cv] if cv is not None else [])+[g[:,j]])
            val = ~np.isnan(X).any(1)
            b_, se = _logistic_irls(X[val], y[val])
            if se is None or se <= 0: valid[j] = False; continue
            z = b_/se; chi2[j] = z**2; pv[j] = float(2*stats.norm.sf(abs(z)))
        valid &= np.isfinite(pv)
    idx = np.where(valid)[0]
    if len(idx) == 0: return pd.DataFrame()
    df = pd.DataFrame({"SNP": snp["SNP"].values[idx], "CHR": snp["CHR"].astype(str).values[idx], "BP": snp["BP"].astype(int).values[idx],
                       "freq_A": pA[idx].round(4), "freq_B": pB[idx].round(4), "delta_freq": (pA-pB)[idx].round(4),
                       "MAF": mj[idx].round(4), "chi2": chi2[idx].round(4), "P": pv[idx], "FST_snp": fv[idx].round(4),
                       "n_A": nA[idx].astype(int), "n_B": nB[idx].astype(int)})
    return df.sort_values("P").reset_index(drop=True)

@cache_data
def gwas_pairwise(gt, snp_df, labels, ra, rb, maf_min=0.05, method="allelic", covs=None):
    return _pairwise_core(gt, snp_df, labels, ra, rb, maf_min, method, covs)

@cache_data
def gwas_pairwise_group(gt, snp_df, groups, ga, gb, maf_min=0.05, method="allelic", covs=None):
    return _pairwise_core(gt, snp_df, groups, ga, gb, maf_min, method, covs)

@cache_data
def get_pairs(labels):
    r = sorted(pd.unique(labels).astype(str))
    return r, [(r[i], r[j]) for i in range(len(r)) for j in range(i+1, len(r))]

def _vc_delta(Y, X0, K):
    n = len(Y); d, U = np.linalg.eigh(K); d = np.maximum(d, 0); Yt = U.T@Y; X0t = U.T@X0
    def nll(dl):
        if dl <= 0: return 1e18
        w = 1.0/np.sqrt(dl*d+1.0); Yw = Yt*w; Xw = X0t*w[:,None]
        try: b, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
        except Exception: return 1e18
        s2 = float(np.sum((Yw-Xw@b)**2)/n)
        return 1e18 if s2 <= 0 else 0.5*(n*np.log(s2)+np.sum(np.log(dl*d+1.0)))
    grid = np.logspace(-3, 3, 30); lls = np.array([nll(dl) for dl in grid]); ib = int(np.argmin(lls))
    try: return max(float(minimize_scalar(nll, bounds=(grid[max(0,ib-1)], grid[min(29,ib+1)]), method="bounded").x), 1e-6)
    except Exception: return max(float(grid[ib]), 1e-6)

_GWAS_COLS = ["SNP_idx","MAF","BETA","SE","T","P"]

@cache_data
def run_gwas(gt, pheno, K, covs, maf_min=0.05, miss_max=0.05, use_lmm=True, is_bin=False):
    pheno = np.asarray(pheno, dtype=float); valid = np.isfinite(pheno)
    if valid.sum() < 10: raise ValueError("<10 individus valides")
    Y = pheno[valid]; gv = gt[valid]; n = len(Y)
    miss = np.isnan(gv).mean(0); Gi = impute_mean(gv); p = Gi.mean(0)/2.0; mj = np.minimum(p, 1-p)
    keep = np.where((miss <= miss_max) & (mj >= maf_min) & (Gi.std(0) > 1e-8))[0]
    if len(keep) == 0: return pd.DataFrame(columns=_GWAS_COLS)
    X0 = np.ones((n,1))
    if covs is not None and np.size(covs) > 0:
        C = np.asarray(covs, dtype=float)[valid]; C = C[:, np.nanstd(C, 0) > 1e-10]; X0 = np.column_stack([X0, C])
    if is_bin:
        rows = []
        for j in keep:
            b_, se = _logistic_irls(np.column_stack([X0, Gi[:,j].astype(float)]), Y)
            if se is None or se <= 0: continue
            tv = b_/se; rows.append({"SNP_idx": int(j), "MAF": float(mj[j]), "BETA": b_, "SE": se, "T": float(tv), "P": float(2*stats.norm.sf(abs(tv)))})
        return pd.DataFrame(rows, columns=_GWAS_COLS)
    lmm = use_lmm and isinstance(K, np.ndarray) and K.shape[0] == gt.shape[0]
    if lmm:
        Kv = K[np.ix_(valid, valid)]; dl = _vc_delta(Y, X0, Kv)
        d, U = np.linalg.eigh(Kv); d = np.maximum(d, 0); w = 1.0/np.sqrt(dl*d+1.0)
        Yt = (U.T@Y)*w; X0t = (U.T@X0)*w[:,None]
    else: Yt, X0t, U, w = Y, X0, None, None
    dof = n-X0t.shape[1]-1
    if dof < 5: raise ValueError("Trop peu d'individus / covariables")
    Q, _ = np.linalg.qr(X0t); Yr = Yt-Q@(Q.T@Yt); yy = float(Yr@Yr); out = []
    for s in range(0, len(keep), 4000):
        idx = keep[s:s+4000]; Gc = Gi[:,idx].astype(np.float64); Gt = (U.T@Gc)*w[:,None] if lmm else Gc
        Gr = Gt-Q@(Q.T@Gt); gg = (Gr**2).sum(0); ok = gg > 1e-10; gg_s = np.where(ok, gg, 1.0)
        beta = (Gr.T@Yr)/gg_s; s2 = np.clip((yy-beta**2*gg_s)/dof, 1e-300, None); se = np.sqrt(s2/gg_s)
        tv = beta/se; pv = 2*t_dist.sf(np.abs(tv), df=dof)
        out.append(pd.DataFrame({"SNP_idx": idx[ok], "MAF": mj[idx][ok], "BETA": beta[ok], "SE": se[ok], "T": tv[ok], "P": pv[ok]}))
    return pd.concat(out, ignore_index=True)

def annotate_gwas(r, snp_df):
    if r.empty: return r
    idx = r["SNP_idx"].astype(int).values; r = r.copy()
    r["SNP"] = snp_df["SNP"].astype(str).values[idx]; r["CHR"] = snp_df["CHR"].astype(str).values[idx]
    r["BP"] = snp_df["BP"].astype(float).astype(np.int64).values[idx]
    r["CM"] = snp_df["CM"].astype(float).values[idx] if "CM" in snp_df.columns else 0.0
    return r[["SNP_idx","SNP","CHR","BP","CM","MAF","BETA","SE","T","P"]]

def lambda_gc(pvals):
    p = np.asarray(pvals); p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10: return np.nan
    return float(np.median(stats.chi2.isf(p, df=1))/stats.chi2.ppf(0.5, df=1))

def _bh(p):
    p = np.asarray(p, dtype=float); n = len(p); o = np.argsort(p)
    q = p[o]*n/np.arange(1, n+1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1); return out

def postprocess_gwas(r, apply_gc=True):
    if r is None or r.empty: return r
    r = r.copy(); r["FDR"] = _bh(r["P"].values); r["log10P"] = -np.log10(r["P"].clip(1e-300, 1))
    if apply_gc:
        lam = lambda_gc(r["P"].values)
        if np.isfinite(lam) and lam > 1.0:
            r["P_GC"] = stats.chi2.sf(stats.chi2.isf(r["P"].clip(1e-300, 1), df=1)/lam, df=1)
            r["log10P_GC"] = -np.log10(r["P_GC"].clip(1e-300, 1))
    return r.sort_values("P").reset_index(drop=True)

def fine_mapping(r, window_mb=1.0, w_prior=0.04):
    if r is None or r.empty: return pd.DataFrame()
    d = r[np.isfinite(r["BETA"]) & np.isfinite(r["SE"]) & (r["SE"] > 0)].copy()
    if d.empty: return d
    lead = d.sort_values("P").iloc[0]
    reg = d[(d["CHR"].astype(str) == str(lead["CHR"])) & ((d["BP"]-lead["BP"]).abs() <= window_mb*1e6)].copy()
    V = reg["SE"]**2; z2 = reg["BETA"]**2/V
    reg["logABF"] = 0.5*np.log(V/(V+w_prior))+0.5*z2*w_prior/(V+w_prior)
    reg["ABF"] = np.exp(reg["logABF"]-reg["logABF"].max()); reg["PIP"] = reg["ABF"]/reg["ABF"].sum()
    reg = reg.sort_values("PIP", ascending=False).reset_index(drop=True); reg["CUM_PIP"] = reg["PIP"].cumsum()
    return reg

def run_gblup(gt, pheno, kf=5, ids=None, seed=42):
    pheno = np.asarray(pheno, dtype=float); valid = np.isfinite(pheno)
    if valid.sum() < 20: raise ValueError("<20 phénotypes")
    Y = pheno[valid]; n = len(Y); kf = max(2, min(kf, n-1))
    X = impute_mean(gt[valid]); p = X.mean(0)/2.0; Z = X-2*p
    G = (Z@Z.T)/(2*np.sum(p*(1-p)))+1e-5*np.eye(n)
    d, U = np.linalg.eigh(G); d = np.maximum(d, 0); mu = Y.mean(); Yt = U.T@(Y-mu)
    def ll(h2):
        lam = (1-h2)/h2; return -0.5*(np.sum(np.log(d+lam))+n*np.log(np.sum(Yt**2/(d+lam))))
    grid = np.linspace(0.01, 0.99, 50); ib = int(np.argmax([ll(h) for h in grid]))
    try: h2 = float(minimize_scalar(lambda h: -ll(h), bounds=(grid[max(0,ib-1)], grid[min(49,ib+1)]), method="bounded").x)
    except Exception: h2 = float(grid[ib])
    lam = (1-h2)/h2; gebv = U@(d/(d+lam)*Yt)
    rng = np.random.default_rng(seed); idx = rng.permutation(n); folds = np.array_split(idx, kf); yp = np.zeros(n)
    for te in folds:
        tr = np.setdiff1d(idx, te); mtr = Y[tr].mean()
        yp[te] = mtr + G[np.ix_(te, tr)]@np.linalg.solve(G[np.ix_(tr, tr)]+lam*np.eye(len(tr)), Y[tr]-mtr)
    corr = float(np.corrcoef(Y, yp)[0,1]) if n > 2 else 0.0
    names = np.asarray(ids)[valid] if ids is not None else np.where(valid)[0]
    return pd.DataFrame({"IID": names, "Phenotype": Y, "GEBV": gebv}), {"h2": h2, "corr": corr, "r2": corr**2, "rmse": float(np.sqrt(np.mean((Y-yp)**2)))}

# ============================================================
# MÉTA & VEP
# ============================================================
def _merge_res(res_race, res_group):
    return res_race[["SNP","CHR","BP","P"]].rename(columns={"P":"P_race"}).merge(res_group[["SNP","P"]].rename(columns={"P":"P_group"}), on="SNP", how="inner")

def combine_pvalues_fisher(res_race, res_group):
    if res_race is None or res_group is None: return None
    m = _merge_res(res_race, res_group)
    if m.empty: return None
    m["P_combined"] = [stats.combine_pvalues(r, method="fisher")[1] for r in m[["P_race","P_group"]].values]
    m["logP_combined"] = -np.log10(m["P_combined"].clip(1e-300, 1))
    return m.sort_values("P_combined").reset_index(drop=True)

def combine_stouffer(res_race, res_group, n_race, n_group):
    if res_race is None or res_group is None: return None
    m = _merge_res(res_race, res_group)
    if m.empty: return None
    wr, wg = np.sqrt(n_race), np.sqrt(n_group)
    zr = stats.norm.isf(m["P_race"].clip(1e-300, 1).values/2); zg = stats.norm.isf(m["P_group"].clip(1e-300, 1).values/2)
    m["P_combined"] = 2*stats.norm.sf(np.abs((wr*zr+wg*zg)/np.sqrt(wr**2+wg**2)))
    m["logP_combined"] = -np.log10(m["P_combined"].clip(1e-300, 1))
    return m.sort_values("P_combined").reset_index(drop=True)

@cache_data
def permutation_test_race(gt, snp_df, labels, ra, rb, n_perm=100, seed=42, max_snp=5000, maf_min=0.05):
    rng = np.random.default_rng(seed); labels = np.asarray(labels).astype(str)
    if len(snp_df) > max_snp:
        idx = np.sort(rng.choice(len(snp_df), max_snp, replace=False)); snp_df = snp_df.iloc[idx].reset_index(drop=True); gt = gt[:,idx]
    original = _pairwise_core(gt, snp_df, labels, ra, rb, maf_min)
    if original.empty: return pd.DataFrame(), 1.0, 0, [], 1.0
    n_obs = int((original["P"] < 0.05/len(original)).sum()); min_obs = original["P"].min()
    mask = np.isin(labels, [ra, rb]); lab_sub = labels[mask].copy(); gt_sub = gt[mask]
    counts, minps = [], []
    for _ in range(n_perm):
        rng.shuffle(lab_sub); r = _pairwise_core(gt_sub, snp_df, lab_sub, ra, rb, maf_min)
        counts.append(int((r["P"] < 0.05/len(r)).sum()) if not r.empty else 0)
        minps.append(float(r["P"].min()) if not r.empty else 1.0)
    return (original, float((np.sum(np.array(counts) >= n_obs)+1)/(n_perm+1)), n_obs, counts,
            float((np.sum(np.array(minps) <= min_obs)+1)/(n_perm+1)))

_SEVERITY = {"HIGH":4, "MODERATE":3, "LOW":2, "MODIFIER":1}

def traduire_consequence_vep(c):
    c = str(c).lower()
    if any(k in c for k in ("stop_gained","frameshift","start_lost","stop_lost")): return "⛔ Exon (Non-sens)"
    if "missense" in c: return "🔴 Exon (Faux-sens)"
    if "synonymous" in c: return "🟡 Exon (Silencieuse)"
    if "splice" in c: return "⚠️ Jonction Exon/Intron"
    if "intron" in c: return "🟢 Intron"
    if "3_prime_utr" in c: return "🔵 3'UTR"
    if "5_prime_utr" in c: return "🔵 5'UTR"
    if "upstream" in c: return "🟣 Promoteur amont"
    if "downstream" in c: return "🟣 Régulateur aval"
    if "regulatory" in c or "tf_binding" in c: return "🟣 Région régulatrice"
    return "⚪ Intergénique"

def run_vep_api(items, species, batch=200):
    if not items: return pd.DataFrame()
    headers = {"Content-Type":"application/json", "Accept":"application/json"}; results = []
    def _post(url, payload, mapping):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
            if r.status_code == 429: time.sleep(float(r.headers.get("Retry-After", 2))); r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
            if r.status_code != 200: st.warning(f"⚠️ VEP HTTP {r.status_code}"); return
            for item in r.json():
                sid = mapping.get(item.get("input"), item.get("input","N/A"))
                if item.get("transcript_consequences"):
                    cons = sorted(item["transcript_consequences"], key=lambda x: _SEVERITY.get(x.get("impact","MODIFIER"), 0), reverse=True)[0]
                    results.append({"SNP": sid, "GENE": cons.get("gene_symbol", cons.get("gene_id","N/A")),
                                    "CONSEQUENCE": ", ".join(cons.get("consequence_terms",["N/A"])), "IMPACT": cons.get("impact","N/A"),
                                    "AMINO_ACID_CHANGE": cons.get("amino_acids","N/A"), "SIFT": cons.get("sift_prediction","N/A"),
                                    "POLYPHEN": cons.get("polyphen_prediction","N/A")})
                elif item.get("regulatory_feature_consequences"):
                    cons = item["regulatory_feature_consequences"][0]
                    results.append({"SNP": sid, "GENE": "Région régulatrice", "CONSEQUENCE": cons.get("consequence_terms",["N/A"])[0],
                                    "IMPACT":"MODIFIER", "AMINO_ACID_CHANGE":"N/A", "SIFT":"N/A", "POLYPHEN":"N/A"})
                else: results.append({"SNP": sid, "GENE":"Intergenic/Non-codant", "CONSEQUENCE": item.get("most_severe_consequence","N/A"),
                                      "IMPACT":"MODIFIER", "AMINO_ACID_CHANGE":"N/A", "SIFT":"N/A", "POLYPHEN":"N/A"})
        except Exception as e: st.warning(f"⚠️ VEP réseau : {e}")
    reg = [(s,v) for s,k,v in items if k == "region"]; ids = [(s,v) for s,k,v in items if k == "id"]
    for i in range(0, len(reg), batch):
        ch = reg[i:i+batch]; _post(f"https://rest.ensembl.org/vep/{species}/region?regulatory=1", {"variants":[v for _,v in ch]}, {v:s for s,v in ch}); time.sleep(0.4)
    for i in range(0, len(ids), batch):
        ch = ids[i:i+batch]; _post(f"https://rest.ensembl.org/vep/{species}/id?regulatory=1", {"ids":[v for _,v in ch]}, {v:s for s,v in ch}); time.sleep(0.4)
    return pd.DataFrame(results)

def fetch_ncbi_gene_info(gene_list, taxon):
    results = []
    for gene in gene_list:
        url = f"https://api.ncbi.nlm.nih.gov/datasets/v2/gene/symbol/{quote(str(gene))}/taxon/{taxon}"
        try:
            r = requests.get(url, timeout=30, headers={"Accept":"application/json"})
            reports = (r.json().get("reports") or []) if r.status_code == 200 else []
            gi = (reports[0].get("gene") or {}) if reports else {}
            if not gi:
                results.append({"GENE": gene, "Nom": "Non trouvé", "Chr": "N/A", "Résumé": "N/A"}); continue
            summ = gi.get("summary")
            if isinstance(summ, list): summ = " ".join(x.get("description","") for x in summ if isinstance(x, dict))
            results.append({"GENE": gene, "Nom": gi.get("description","N/A"), "Chr": (gi.get("chromosomes") or ["N/A"])[0], "Résumé": (str(summ)[:200]+"...") if summ else "N/A"})
        except Exception as e:
            results.append({"GENE": gene, "Nom": "Erreur API", "Chr": "N/A", "Résumé": str(e)})
        time.sleep(0.4)
    return pd.DataFrame(results)

def fetch_uniprot_id(gene_name, taxon):
    q = f"gene:{gene_name} AND organism_id:{taxon}"
    try:
        r = requests.get(f"https://rest.uniprot.org/uniprotkb/search?query={quote(q)}&format=json&size=1", timeout=30)
        if r.status_code == 200 and r.json().get("results"): return r.json()["results"][0].get("primaryAccession")
    except Exception: pass
    return None

def fetch_alphafold_pdb(uniprot_id):
    try:
        r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}", timeout=30)
        if r.status_code == 200:
            recs = r.json(); rec = next((x for x in recs if x.get("uniprotAccession") == uniprot_id), recs[0] if recs else None)
            if rec and rec.get("pdbUrl"):
                pr = requests.get(rec["pdbUrl"], timeout=30)
                if pr.status_code == 200: return pr.text
    except Exception: pass
    return None

def render_3d_structure(pdb_data, style="cartoon", color_scheme="spectrum", highlight_res=None):
    sty = {style: ({"color":"spectrum"} if color_scheme == "spectrum" else {"colorscheme": color_scheme})}; hl = ""
    if highlight_res and str(highlight_res).strip().isdigit():
        hl = f"viewer.addStyle({{resi: {int(highlight_res)}}}, {{stick: {{colorscheme: 'redCarbon'}}}});"
    html = f"""<div id="viewport" style="width:100%;height:500px;position:relative;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      let viewer = $3Dmol.createViewer(document.getElementById("viewport"), {{backgroundColor: 'white'}});
      viewer.addModel({json.dumps(pdb_data)}, "pdb"); viewer.setStyle({{}}, {json.dumps(sty)}); {hl}
      viewer.zoomTo(); viewer.render();
    </script>"""
    components.html(html, height=520)

def fetch_ensembl_sequence(chrom, start, end, species):
    try:
        r = requests.get(f"https://rest.ensembl.org/sequence/region/{species}/{chrom}:{start}..{end}:1?content-type=text/plain", timeout=30)
        if r.status_code == 200: return r.text.strip().upper()
    except Exception: pass
    return None

def align_and_highlight(seq_ref, seq_a, seq_b, name_a="Allèle A", name_b="Allèle B"):
    if not HAS_BIO: return "<p>⚠️ Biopython non installé.</p>"
    al = Align.PairwiseAligner(); al.mode = "global"; al.match_score = 2; al.mismatch_score = -3; al.open_gap_score = -5; al.extend_gap_score = -2
    def fmt(seq, name, color):
        aln = al.align(seq_ref, seq)[0]; ref_aln, seq_aln = str(aln[0]), str(aln[1]); s = f"<b>{name:10}</b> | "
        for r, q in zip(ref_aln, seq_aln):
            if r == "-" or q == "-": s += f"<span style='color:orange;font-weight:bold;'>{q}</span>"
            elif r != q: s += f"<span style='color:red;font-weight:bold;'>{q}</span>"
            else: s += f"<span style='color:{color};'>{q}</span>"
        return s+"<br>", ref_aln
    ha, ref_aln = fmt(seq_a, name_a, "green"); hb, _ = fmt(seq_b, name_b, "blue")
    return "<div style='font-family:monospace;font-size:16px;line-height:1.5;'>"+ha+f"<b>{'Référence':10}</b> | "+"".join(f"<span style='color:grey;'>{r}</span>" for r in ref_aln)+"<br>"+hb+"</div>"

def ortholog_lookup(chrom, bp, species, target="homo_sapiens"):
    base = "https://rest.ensembl.org"; h = {"Content-Type":"application/json","Accept":"application/json"}
    try:
        genes = []
        for w in (0, 50000):
            r = requests.get(f"{base}/overlap/region/{species}/{chrom}:{max(1,bp-w)}-{bp+w}?feature=gene", headers=h, timeout=30)
            genes = r.json() if r.status_code == 200 else []
            if genes: break
        if not genes: return None
        g = next((x for x in genes if x.get("biotype") == "protein_coding"), genes[0]); gid = g["id"]
        r = requests.get(f"{base}/homology/id/{species}/{gid}?target_species={target};type=orthologues", headers=h, timeout=60)
        data = r.json().get("data", []) if r.status_code == 200 else []
        homs = data[0].get("homologies", []) if data else []
        if not homs: return {"gene_src": g.get("external_name", gid), "gene_id": gid, "human_gene": None}
        t = homs[0]["target"]
        r = requests.get(f"{base}/lookup/id/{t['id']}", headers=h, timeout=30); info = r.json() if r.status_code == 200 else {}
        return {"gene_src": g.get("external_name", gid), "gene_id": gid, "human_gene": info.get("display_name", t["id"]),
                "human_id": t["id"], "chr": info.get("seq_region_name"), "start": info.get("start"), "end": info.get("end"),
                "perc_id": t.get("perc_id"), "assembly": info.get("assembly_name")}
    except Exception: return None

def annotate_genes(snp_df, gff, wk=50):
    if gff is None or gff.empty: return pd.DataFrame()
    w = wk*1000; res = []; gchr = gff["CHR"].astype(str).values
    for _, s in snp_df.iterrows():
        pos = int(s.get("BP", s.get("START", 0))); ch = str(s["CHR"])
        ng = gff[(gchr == ch) & (gff["START"] <= pos+w) & (gff["END"] >= pos-w)]
        for _, g in ng.iterrows():
            d = 0 if g["START"] <= pos <= g["END"] else min(abs(g["START"]-pos), abs(g["END"]-pos))
            res.append({"SNP": s.get("SNP", f"{ch}:{pos}"), "CHR": ch, "SNP_POS": pos, "GENE": g["GENE"],
                        "GENE_START": g["START"], "GENE_END": g["END"], "DIST_KB": round(d/1000.0, 2)})
    return pd.DataFrame(res)

def run_enrichment(gene_list, gene_sets=None, organism="Human", outdir=None):
    try:
        import gseapy as gp
        genes = list({str(g).strip() for g in gene_list if str(g).strip() and str(g) != "nan"})
        if len(genes) < 3: return pd.DataFrame()
        if gene_sets is None: gene_sets = ["GO_Biological_Process_2023","KEGG_2021_Human"]
        enr = gp.enrichr(gene_list=genes, gene_sets=gene_sets, organism=organism, outdir=outdir, cutoff=1.0)
        if enr is None or enr.results is None: return pd.DataFrame()
        df = enr.results.copy()
        for old, new in {"Term":"TERM","Adjusted P-value":"FDR","P-value":"PVAL","Overlap":"OVERLAP","Genes":"GENES","Combined Score":"SCORE"}.items():
            if old in df.columns: df = df.rename(columns={old: new})
        return df.sort_values("FDR").reset_index(drop=True) if "FDR" in df.columns else df
    except Exception as e:
        st.error(f"❌ Enrichr : {e}"); return pd.DataFrame()

def query_string_network(genes, species, required_score=400):
    if not genes or len(genes) < 2: return pd.DataFrame(), pd.DataFrame()
    ids = "%0d".join([quote(str(g).strip()) for g in genes if str(g).strip()])
    url = f"https://string-db.org/api/json/network?identifiers={ids}&species={species}&required_score={required_score}&caller_identity=livestock_genomics_pipeline"
    try:
        r = requests.get(url, timeout=60); r.raise_for_status(); data = r.json()
        if not data: return pd.DataFrame(), pd.DataFrame()
        edges = pd.DataFrame(data)[["preferredName_A","preferredName_B","score"]].drop_duplicates()
        edges.columns = ["SOURCE","TARGET","SCORE"]
        return edges, pd.DataFrame({"ID": sorted(set(edges["SOURCE"]).union(set(edges["TARGET"])))})
    except Exception: return pd.DataFrame(), pd.DataFrame()

# ============================================================
# EXPORTS
# ============================================================
def _alleles_for_export(snp_df, m):
    a_minor = snp_df["A2"].astype(str).values[:m] if "A2" in snp_df.columns else np.array(["G"]*m)
    a_major = snp_df["A1"].astype(str).values[:m] if "A1" in snp_df.columns else np.array(["A"]*m)
    return a_minor, a_major

def build_plink(gt, ind_df, snp_df, prefix="genome"):
    n, m = gt.shape; nb = (n+3)//4
    code = np.full(gt.shape, 0b01, dtype=np.uint8)
    code[gt == 0] = 0b11; code[gt == 1] = 0b10; code[gt == 2] = 0b00
    code = np.vstack([code, np.zeros((nb*4-n, m), dtype=np.uint8)]).reshape(nb, 4, m)
    packed = code[:,0,:] | (code[:,1,:]<<2) | (code[:,2,:]<<4) | (code[:,3,:]<<6)
    a1, a2 = _alleles_for_export(snp_df, m)
    clean = lambda a: np.where(np.isin(a, ["",".","nan","None"]), "0", a)
    bim = pd.DataFrame({"c": snp_df["CHR"].values[:m], "s": snp_df["SNP"].values[:m], "cm": snp_df["CM"].values[:m],
                        "bp": snp_df["BP"].astype(int).values[:m], "a1": clean(a1), "a2": clean(a2)})
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}.bed", bytes([0x6C,0x1B,0x01])+np.ascontiguousarray(packed.T).tobytes())
        zf.writestr(f"{prefix}.bim", bim.to_csv(sep="\t", header=False, index=False))
        zf.writestr(f"{prefix}.fam", "\n".join(f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9" for _, r in ind_df.iterrows())+"\n")
    return bio.getvalue()

def build_vcf(gt, ind_df, snp_df, proj="Genome"):
    m = gt.shape[1]; alt, ref = _alleles_for_export(snp_df, m)
    acgt = np.array(["A","C","G","T"]); ref = np.where(np.isin(ref, acgt), ref, "A"); alt = np.where(np.isin(alt, acgt), alt, "G")
    h = ["##fileformat=VCFv4.2", f"##source={proj}", '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
         "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"+"\t".join(ind_df["IID"].astype(str))]
    lut = np.array(["0/0","0/1","1/1"])
    strs = np.where(np.isnan(gt), "./.", lut[np.nan_to_num(gt, nan=0).astype(int).clip(0,2)])
    b = []
    for j in range(m):
        r = snp_df.iloc[j]; b.append(f"{r['CHR']}\t{int(r['BP'])}\t{r['SNP']}\t{ref[j]}\t{alt[j]}\t.\tPASS\t.\tGT\t"+"\t".join(strs[:,j]))
    return "\n".join(h+b)+"\n"

# ============================================================
# PLOT FUNCTIONS
# ============================================================
def _cum_positions(df, poscol):
    df = df.assign(_k=df["CHR"].map(_chk_num), _s=df["CHR"].astype(str)).sort_values(["_k","_s",poscol], kind="stable").reset_index(drop=True)
    offs = {}; ticks = []; cur = 0.0
    for c, sub in df.groupby("CHR", sort=False):
        mx = float(sub[poscol].max())/1e6; offs[c] = cur; ticks.append((str(c), cur+mx/2)); cur += mx+5
    df["_x"] = df["CHR"].map(offs)+df[poscol].astype(float)/1e6
    return df, ticks

def plot_missing(mi, ms):
    f = make_subplots(rows=1, cols=2, subplot_titles=("Missing/Ind","Missing/SNP"))
    f.add_trace(go.Histogram(x=mi, nbinsx=60, marker_color="skyblue"), 1, 1)
    f.add_trace(go.Histogram(x=ms, nbinsx=60, marker_color="coral"), 1, 2)
    f.update_layout(height=400, showlegend=False); return f

def plot_pca(s, v, l):
    return px.scatter(pd.DataFrame({"PC1":s[:,0],"PC2":s[:,1],"Pop":l}), x="PC1", y="PC2", color="Pop",
                      title=f"PCA PC1({v[0]:.1f}%) vs PC2({v[1]:.1f}%)", height=550).update_traces(marker=dict(size=10, line=dict(width=1, color="white")))

def plot_mds(m, l):
    return px.scatter(pd.DataFrame({"MDS1":m[:,0],"MDS2":m[:,1],"Pop":list(l)}), x="MDS1", y="MDS2", color="Pop",
                      title="MDS", height=520).update_traces(marker=dict(size=11, line=dict(width=1, color="white")))

def plot_manhattan(df, yc="P", title="", thr=None):
    df = df.copy(); ylab = yc
    if yc in ("P","P_GC"): df["_y"] = -np.log10(np.clip(df[yc], 1e-300, 1)); yc = "_y"; ylab = "-log10(P)"
    df = df[np.isfinite(df[yc])]
    if df.empty: return go.Figure()
    df, ticks = _cum_positions(df, "BP"); fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR", sort=False)):
        fig.add_trace(go.Scatter(x=sub["_x"], y=sub[yc], mode="markers",
                                 marker=dict(size=5, color="#2c3e50" if i%2 == 0 else "#7f8c8d"),
                                 text=sub["SNP"], hoverinfo="text+y", showlegend=False))
    if thr: fig.add_hline(y=thr, line_dash="dash", line_color="red")
    fig.update_layout(title=title, xaxis=dict(title="Chromosome", tickmode="array", tickvals=[t[1] for t in ticks], ticktext=[t[0] for t in ticks]),
                      yaxis_title=ylab, height=500); return fig

def plot_qq(pvals, title="QQ-plot"):
    p = np.sort(np.asarray(pvals)[np.isfinite(pvals)]); p = p[(p > 0) & (p <= 1)]; n = len(p)
    if n == 0: return go.Figure()
    exp = -np.log10((np.arange(1, n+1)-0.5)/n); obs = -np.log10(p)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=exp, y=obs, mode="markers", marker=dict(size=4), name="obs"))
    fig.add_trace(go.Scatter(x=[0, exp.max()], y=[0, exp.max()], mode="lines", line=dict(color="red", dash="dash"), name="attendu"))
    fig.update_layout(title=title, xaxis_title="-log10(P) attendu", yaxis_title="-log10(P) observé", height=420)
    return fig

def plot_ld(ld):
    if ld is None or ld.empty: return None
    return px.line(ld.sort_values("dist_kb"), x="dist_kb", y="r2", markers=True,
                   title="Décroissance du LD (r² moyen)", labels={"dist_kb":"Distance (kb)","r2":"r²"}, height=450)

def plot_admix(Q, labs, pops, K):
    df = pd.DataFrame(Q, columns=[f"K{k+1}" for k in range(K)]); df["IID"] = labs; df["Pop"] = pops; df["_d"] = Q.argmax(1)
    df["_po"] = pd.Categorical(pops, categories=sorted(set(pops)))
    df = df.sort_values(["_po","_d"]).reset_index(drop=True); df["x"] = np.arange(len(df)); fig = go.Figure()
    for k in range(K):
        fig.add_trace(go.Bar(x=df["x"], y=df[f"K{k+1}"], name=f"K{k+1}", marker_color=px.colors.qualitative.Set2[k%8]))
    fig.update_layout(barmode="stack", title=f"Admixture (NMF) K={K}", height=500)
    return fig, df

def plot_gblup(df):
    fig = px.scatter(df, x="Phenotype", y="GEBV", title="GBLUP")
    fig.update_traces(marker=dict(size=10, color="royalblue"))
    if len(df) > 1:
        try:
            z = np.polyfit(df["Phenotype"], df["GEBV"], 1); xr = np.linspace(df["Phenotype"].min(), df["Phenotype"].max(), 100)
            fig.add_trace(go.Scatter(x=xr, y=np.poly1d(z)(xr), mode="lines", name="OLS", line=dict(color="red", dash="dash")))
        except Exception: pass
    return fig

def plot_pip(df):
    return px.bar(df.head(20), x="SNP", y="PIP", title="PIP (probabilité d'inclusion)", color="PIP", color_continuous_scale="Viridis")

def plot_win_manh(df, vc="MEAN_FST", title="", sigma=3.0):
    df = df[np.isfinite(df[vc])].copy()
    if df.empty: return None
    df, ticks = _cum_positions(df, "MID"); fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR", sort=False)):
        fig.add_trace(go.Scatter(x=sub["_x"], y=sub[vc], mode="markers",
                                 marker=dict(size=7, color="#2c3e50" if i%2 == 0 else "#7f8c8d", line=dict(width=0.5, color="white")),
                                 text=[f"{c}:{int(m/1e6)}Mb<br>{vc}={v:.4f}" for m, v in zip(sub["MID"], sub[vc])],
                                 hoverinfo="text", showlegend=False))
    mu, sd = df[vc].mean(), df[vc].std()
    fig.add_hline(y=mu+sigma*sd, line_dash="dash", line_color="red", annotation_text=f"μ+{sigma}σ")
    fig.update_layout(title=title, xaxis=dict(title="Chromosome", tickmode="array", tickvals=[t[1] for t in ticks], ticktext=[t[0] for t in ticks]),
                      yaxis_title=vc, height=520); return fig

def plot_gene_track(si, gff, wk=50):
    if gff is None or gff.empty: return None
    ch = str(si["CHR"]); pos = int(si.get("BP", si.get("START", 0))); w = wk*1000
    rg = gff[(gff["CHR"].astype(str) == ch) & (gff["START"] <= pos+w) & (gff["END"] >= pos-w)]
    fig = go.Figure()
    for i, (_, g) in enumerate(rg.iterrows()):
        fig.add_trace(go.Scatter(x=[g["START"]/1e6, g["END"]/1e6], y=[i,i], mode="lines",
                                 line=dict(width=10, color="royalblue"), name=g["GENE"], text=[g["GENE"]]*2, hoverinfo="text"))
    fig.add_vline(x=pos/1e6, line_dash="dash", line_color="red", annotation_text=f"SNP: {si.get('SNP','')}")
    fig.update_layout(title=f"Gènes CHR{ch}", xaxis_title="Pos (Mb)", yaxis=dict(showticklabels=False), height=400+len(rg)*20)
    return fig

def plot_string_network(edges, nodes):
    if edges is None or edges.empty: return None
    try: import networkx as nx
    except ImportError: st.error("⚠️ `networkx` non installé."); return None
    G = nx.Graph()
    for _, r in edges.iterrows(): G.add_edge(r["SOURCE"], r["TARGET"], weight=float(r["SCORE"]))
    pos = nx.spring_layout(G, seed=42, k=0.6); ex, ey = [], []
    for s, t in G.edges():
        x0, y0 = pos[s]; x1, y1 = pos[t]; ex += [x0, x1, None]; ey += [y0, y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.6, color="#cccccc"), hoverinfo="skip", showlegend=False))
    deg = [d for _, d in G.degree()]
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
                             mode="markers+text", marker=dict(size=[8+3*d for d in deg], color=deg, colorscale="Viridis", line=dict(width=1, color="white")),
                             text=list(G.nodes()), textposition="top center", hoverinfo="text"))
    fig.update_layout(showlegend=False, height=600, xaxis=dict(showgrid=False, zeroline=False, visible=False), yaxis=dict(showgrid=False, zeroline=False, visible=False))
    return fig

def plot_tree(Z, labels, support=None, title="", mid=None):
    n = len(labels); labels = [str(l) for l in labels]
    lo = leaves_list(Z); ly = {int(l): i for i, l in enumerate(lo)}; cy = {i: ly[i] for i in range(n)}; cx = {i: 0.0 for i in range(n)}
    fig = go.Figure()
    for k, row in enumerate(Z):
        i, j, d = int(row[0]), int(row[1]), float(row[2]); yi, yj = cy[i], cy[j]; xi, xj = cx[i], cx[j]
        if support is not None and k < len(support):
            s = support[k]; col = "#2ecc71" if s >= 70 else "#f39c12" if s >= 50 else "#e74c3c"
        else: col = "#3498db"
        for xs, ys in (([xi,d],[yi,yi]), ([xj,d],[yj,yj]), ([d,d],[yi,yj])):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=2), hoverinfo="skip", showlegend=False))
        cy[n+k] = (yi+yj)/2; cx[n+k] = d
        if support is not None and k < len(support):
            fig.add_annotation(x=d, y=(yi+yj)/2, text=f"<b>{int(support[k])}</b>", showarrow=False,
                               font=dict(size=10, color=col), yshift=8, bgcolor="white", borderpad=2)
    for leaf, y in ly.items():
        fig.add_annotation(x=0, y=y, text=f"<b>{labels[leaf]}</b>", showarrow=False, xanchor="right", xshift=-8,
                           font=dict(size=12, color="#2c3e50"))
    if mid: fig.add_vline(x=mid["midpoint"], line_dash="dot", line_color="purple")
    fig.update_layout(title=title, xaxis_title="Distance de Reynolds",
                      yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                      height=max(500, n*35), margin=dict(l=120, r=40, t=80, b=40), plot_bgcolor="white")
    return fig

def plot_snp_density(snp_df, window_kb=1000):
    if snp_df is None or snp_df.empty: return go.Figure()
    df = snp_df.copy(); df["BP"] = df["BP"].astype(int); df["window"] = (df["BP"]//(window_kb*1000)).astype(int)
    density = df.groupby(["CHR","window"]).size().reset_index(name="count")
    chrs = sorted(df["CHR"].unique(), key=lambda c: int(c) if str(c).isdigit() else 100)
    chrs_y = {c: i for i, c in enumerate(chrs)}
    grid = []
    for c in chrs:
        max_w = density[density["CHR"] == c]["window"].max() if not density[density["CHR"] == c].empty else 0
        c_dens = density[density["CHR"] == c].set_index("window")["count"]
        for w in range(max_w+1): grid.append({"CHR": c, "window": w, "count": c_dens.get(w, 0)})
    grid = pd.DataFrame(grid)
    cs = [[0, 'white'], [0.05, 'darkgreen'], [0.2, 'green'], [0.5, 'yellow'], [0.8, 'orange'], [1.0, 'red']]
    fig = go.Figure(data=go.Heatmap(x=grid["window"]*window_kb*1000/1e6, y=grid["CHR"].map(chrs_y), z=grid["count"],
                                    colorscale=cs, zmin=0, hoverongaps=False, colorbar=dict(title="Nb SNPs")))
    fig.update_layout(title=f"Nombre de SNPs par fenêtre de {window_kb} kb", xaxis_title="Position (Mb)",
                      yaxis=dict(tickvals=list(chrs_y.values()), ticktext=list(chrs_y.keys()), title="Chromosome"), height=600)
    return fig

def plot_shannon(shannon_df):
    if shannon_df.empty: return go.Figure()
    df, ticks = _cum_positions(shannon_df, "MID"); fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR", sort=False)):
        fig.add_trace(go.Scatter(x=sub["_x"], y=sub["SHANNON_H"], mode="lines+markers", name=f"CHR {c}",
                                 marker=dict(size=4), line=dict(width=1)))
    fig.update_layout(title="Indice de Diversité de Shannon", xaxis=dict(title="Chromosome", tickmode="array",
                      tickvals=[t[1] for t in ticks], ticktext=[t[0] for t in ticks]),
                      yaxis_title="Shannon H", height=500, showlegend=False)
    return fig

def plot_ne(ne_df):
    if ne_df.empty: return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ne_df["GenAgo"], y=ne_df["Ne"], mode="lines+markers",
                             marker=dict(color="#00bfff", size=8, line=dict(width=1, color="white")), line=dict(width=2)))
    fig.update_layout(title="Taille Efficace de la Population (Ne) historique", xaxis_title="Générations dans le passé",
                      yaxis_title="Ne", xaxis_type="log", yaxis_type="log", height=500)
    return fig

def plot_roh_summary(roh_df):
    if roh_df is None or roh_df.empty: return go.Figure()
    fig = px.histogram(roh_df, x="length_kb", nbins=50, title="Distribution longueur ROH (kb)",
                       labels={"length_kb":"Longueur (kb)"}, height=450)
    return fig

def plot_tajima(td_df):
    if td_df.empty: return go.Figure()
    df, ticks = _cum_positions(td_df, "MID"); fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR", sort=False)):
        fig.add_trace(go.Scatter(x=sub["_x"], y=sub["TAJIMA_D"], mode="markers",
                                 marker=dict(size=5, color="#2c3e50" if i%2 == 0 else "#7f8c8d"),
                                 text=sub["CHR"], hoverinfo="text+y", showlegend=False))
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="Seuil -2")
    fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Seuil +2")
    fig.update_layout(title="Tajima's D scan génomique", xaxis=dict(title="Chromosome", tickmode="array",
                      tickvals=[t[1] for t in ticks], ticktext=[t[0] for t in ticks]),
                      yaxis_title="Tajima's D", height=500)
    return fig

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.title("🧬 Helixa | Universal Livestock Genomics Pipeline v24.0")
    st.markdown("**Bovin & Camelin** · Analyses génomiques pour publication")

    with st.sidebar.expander("🔍 État du LD pruning", expanded=False):
        st.markdown("""
        **✅ Appliqué automatiquement :**
        - PCA · Admixture (NMF) · MDS
        - Arbre UPGMA (Reynolds) · Fst globale

        **⛔ Jamais appliqué :**
        - GWAS · Fine-mapping
        - Tajima's D · Fst par SNP
        - ROH · LD decay
        - Diversité · QC

        *Paramètres : r²=0.2, w=50, step=5, autosomes.*
        """)

    # ---- CHARGEMENT ----
    species = st.sidebar.selectbox("Espèce", list(SPECIES_CONFIG.keys()), key="species")
    use_demo = st.sidebar.checkbox("Mode démo (données simulées)", value=DEMO, key="use_demo")

    ped_file = map_file = pheno_file = gff_file = None
    if not use_demo:
        ped_file = st.sidebar.file_uploader("Fichier PED", type=["ped","txt","gz"], key="ped")
        map_file = st.sidebar.file_uploader("Fichier MAP", type=["map","txt","gz"], key="map")
        pheno_file = st.sidebar.file_uploader("Phénotypes (optionnel)", type=["csv","txt"], key="pheno")
        gff_file = st.sidebar.file_uploader("GFF3 (optionnel)", type=["gff3","gff","txt","gz"], key="gff")

    if not use_demo and (ped_file is None or map_file is None):
        st.info("👈 Importez PED + MAP, ou activez le mode démo.")
        st.stop()

    # ---- LOAD ----
    with st.spinner("Chargement des données..."):
        if use_demo:
            gt, ind_df, snp_df = gen_demo(n_ind=150, n_snp=2000, n_pop=4,
                                          autosomes=SPECIES_CONFIG[species]["autosomes"],
                                          ars=SPECIES_CONFIG[species]["ars"])
            pheno_df = pd.DataFrame({"IID": ind_df["IID"], "Lait": np.random.default_rng(7).normal(20, 4, len(ind_df))})
        else:
            snp_df = parse_map(map_file.getvalue())
            gt, ind_df, a1, a2 = parse_ped(ped_file.getvalue(), n_map=len(snp_df))
            snp_df["A1"], snp_df["A2"] = a1, a2
            pheno_df = load_pheno(pheno_file.getvalue()) if pheno_file else None

        gff = None
        if gff_file:
            try: gff = parse_gff3(gff_file.getvalue())
            except Exception as e: st.sidebar.warning(f"GFF ignoré : {e}")

    # ---- QC PARAMS ----
    with st.sidebar.expander("⚙️ Paramètres QC", expanded=True):
        maf_thr = st.slider("MAF min", 0.0, 0.5, 0.05, 0.01, key="maf")
        geno_thr = st.slider("Missing SNP max", 0.0, 1.0, 0.10, 0.01, key="geno")
        mind_thr = st.slider("Missing Ind max", 0.0, 1.0, 0.20, 0.01, key="mind")
        hwe_thr = st.number_input("HWE p-value min", 0.0, 1.0, 1e-6, format="%.1e", key="hwe")

    try:
        gt_qc, ind_qc, snp_qc, qc_rep = apply_qc(
            gt, ind_df, snp_df,
            {"maf": maf_thr, "geno": geno_thr, "mind": mind_thr, "hwe": hwe_thr, "het_sd": 3.0},
            SPECIES_CONFIG[species]["autosomes"])
        st.sidebar.success(f"✅ QC : {gt_qc.shape[0]} ind × {gt_qc.shape[1]:,} SNPs")
    except Exception as e:
        st.error(f"❌ QC : {e}"); st.stop()

    labels = ind_qc["FID"].astype(str).values
    autosomes = SPECIES_CONFIG[species]["autosomes"]

    # ---- TABS ----
    (t_qc, t_struct, t_demo, t_gwas, t_select, t_annot, t_export) = st.tabs([
        "📊 QC & Diversité", "🧬 Structure", "🌊 LD & Démographie",
        "🧪 GWAS & Prédiction", "🌿 Sélection & Fst scan", "📖 Annotation", "💾 Export"
    ])

    # ==== QC & DIVERSITÉ ====
    with t_qc:
        st.header("📊 Qualité & Diversité")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Individus", gt_qc.shape[0])
            st.metric("SNPs", f"{gt_qc.shape[1]:,}")
        with c2:
            st.metric("MAF moyen", f"{np.nanmean(maf(gt_qc)):.3f}")
            st.metric("Missing moyen", f"{np.nanmean(miss_snp(gt_qc)):.4f}")

        st.subheader("📈 Distribution du missing")
        st.plotly_chart(plot_missing(miss_ind(gt_qc), miss_snp(gt_qc)), use_container_width=True)

        st.subheader("🗺️ Carte de densité des SNPs")
        win_kb = st.select_slider("Fenêtre densité (kb)", options=[100,250,500,1000,5000], value=1000, key="dens_win")
        st.plotly_chart(plot_snp_density(snp_qc, window_kb=win_kb), use_container_width=True)

        st.subheader("📊 Tableau de diversité par population")
        div = compute_diversity(gt_qc, labels)
        if not div.empty: st.dataframe(div, use_container_width=True)

        st.subheader("🌿 Diversité de Shannon")
        win_sh = st.select_slider("Fenêtre Shannon (kb)", options=[100,250,500,1000], value=500, key="sh_win")
        if st.button("Calculer Shannon", key="btn_sh"):
            with st.spinner("Calcul..."):
                st.session_state["sh_df"] = shannon_scan_phys(gt_qc, snp_qc, window_kb=win_sh, step_kb=win_sh//2)
        if "sh_df" in st.session_state:
            st.plotly_chart(plot_shannon(st.session_state["sh_df"]), use_container_width=True)

    # ==== STRUCTURE (AVEC LD) ====
    with t_struct:
        st.header("🧬 Structure des Populations")
        gt_ld, snp_ld = view_ld_pruned(gt_qc, snp_qc, autosomes, r2=0.2)
        caption_ld(gt_qc, gt_ld, r2=0.2)

        s1, s2, s3, s4 = st.tabs(["PCA","Admixture","MDS","Arbre UPGMA"])

        with s1:
            pc, var = pca_analysis(gt_ld, nc=5)
            st.plotly_chart(plot_pca(pc, var, labels), use_container_width=True)

        with s2:
            if "admix_K" not in st.session_state or not isinstance(st.session_state["admix_K"], int):
                st.session_state["admix_K"] = 4
            _v = st.session_state["admix_K"]
            if not (2 <= _v <= 10): _v = 4
            st.session_state["admix_K"] = _v
            K = st.slider("K (composantes)", 2, 10, key="admix_K")
            Q, _H = admixture_nmf(gt_ld, K=K)
            fig, _ = plot_admix(Q, ind_qc["IID"].values, labels, K)
            st.plotly_chart(fig, use_container_width=True)

        with s3:
            mds = mds_analysis(gt_ld, nc=2)
            st.plotly_chart(plot_mds(mds, labels), use_container_width=True)

        with s4:
            if st.button("Construire l'arbre UPGMA", key="btn_tree"):
                with st.spinner("Bootstrap en cours..."):
                    st.session_state["tree"] = upgma_bootstrap(gt_ld, snp_ld, labels, n_boot=100)
            if "tree" in st.session_state:
                tr = st.session_state["tree"]
                st.plotly_chart(plot_tree(tr["Z"], tr["pops"], support=tr["support"], mid=midpoint_info(tr["D"], tr["pops"])), use_container_width=True)
                st.code(tr["newick"], language="text")

        st.subheader("Fst globale (par paire)")
        M_fst, pops = fst_pairwise(gt_ld, labels)
        st.dataframe(pd.DataFrame(M_fst, index=pops, columns=pops).style.background_gradient(cmap="YlOrRd"), use_container_width=True)

    # ==== LD & DÉMOGRAPHIE (SANS LD) ====
    with t_demo:
        st.header("🌊 LD & Démographie")
        st.caption("ℹ️ Pas de LD pruning — analyse qui mesure directement le LD ou la densité.")

        with st.expander("📉 Décroissance du LD", expanded=True):
            ld_df = ld_decay(gt_qc, snp_qc, max_kb=1000)
            st.plotly_chart(plot_ld(ld_df), use_container_width=True)

        with st.expander("📈 Ne historique", expanded=True):
            if st.button("Estimer Ne", key="btn_ne"):
                with st.spinner("Estimation..."):
                    st.session_state["ne_df"] = ne_historical(gt_qc, snp_qc)
            if "ne_df" in st.session_state:
                st.plotly_chart(plot_ne(st.session_state["ne_df"]), use_container_width=True)
                st.dataframe(st.session_state["ne_df"], use_container_width=True)

        with st.expander("🧩 ROH (Runs of Homozygosity)", expanded=False):
            if st.button("Détecter ROH", key="btn_roh"):
                with st.spinner("Détection..."):
                    roh, fb = detect_roh(gt_qc, snp_qc["CHR"].values, snp_qc["BP"].values)
                    st.session_state["roh"] = (roh, fb)
            if "roh" in st.session_state:
                roh, fb = st.session_state["roh"]
                st.write(f"**{len(roh)} ROH détectés**")
                if not roh.empty: st.dataframe(roh.head(100), use_container_width=True)
                st.plotly_chart(plot_roh_summary(roh), use_container_width=True)

    # ==== GWAS & PRÉDICTION ====
    with t_gwas:
        st.header("🧪 GWAS & Prédiction génomique")
        st.caption("ℹ️ Pas de LD pruning — la GRM (kinship) gère la structure de LD.")

        if pheno_file is None and not use_demo:
            st.warning("Importez un fichier de phénotypes.")
        else:
            try:
                matched, id_c = match_pheno(ind_qc, pheno_df)
                pheno_cols = [c for c in matched.columns if c not in ("FID","IID") and pd.api.types.is_numeric_dtype(matched[c])]
                if not pheno_cols:
                    st.error("Aucune colonne phénotypique numérique.")
                else:
                    trait = st.selectbox("Phénotype", pheno_cols, key="trait")
                    gwas_mode = st.radio("Méthode", ["LMM (GRM)", "Pairwise allélique"], horizontal=True, key="gwas_mode")

                    if gwas_mode == "LMM (GRM)":
                        K = grm_vanraden(gt_qc)
                        if st.button("Lancer GWAS (LMM)", key="btn_gwas"):
                            with st.spinner("GWAS LMM..."):
                                res = run_gwas(gt_qc, matched[trait].values, K, None, use_lmm=True)
                                res = annotate_gwas(res, snp_qc); res = postprocess_gwas(res)
                                st.session_state["gwas_res"] = res
                    else:
                        pairs_avail, pairs = get_pairs(labels)
                        pa, pb = st.selectbox("Pop A", pairs_avail, key="pa"), st.selectbox("Pop B", pairs_avail, key="pb")
                        if pa != pb and st.button("Lancer GWAS pairwise", key="btn_gwas_p"):
                            with st.spinner("GWAS pairwise..."):
                                res = gwas_pairwise(gt_qc, snp_qc, labels, pa, pb)
                                if not res.empty:
                                    st.session_state["gwas_res"] = postprocess_gwas(res)

                    if "gwas_res" in st.session_state and not st.session_state["gwas_res"].empty:
                        res = st.session_state["gwas_res"]
                        st.subheader("Manhattan")
                        thr = -np.log10(0.05/max(len(res),1))
                        st.plotly_chart(plot_manhattan(res, "P", "GWAS", thr=thr), use_container_width=True)
                        st.subheader("QQ-plot")
                        st.plotly_chart(plot_qq(res["P"].values), use_container_width=True)
                        st.write(f"**λ GC** : {lambda_gc(res['P'].values):.3f}")
                        st.dataframe(res.head(100), use_container_width=True)

                        if st.button("Fine-mapping (ABF)", key="btn_fm"):
                            fm = fine_mapping(res, window_mb=1.0)
                            st.plotly_chart(plot_pip(fm), use_container_width=True)
                            st.dataframe(fm.head(20), use_container_width=True)

                        csv = res.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇️ Télécharger GWAS (CSV)", csv, "gwas.csv", "text/csv")

                    st.markdown("---")
                    st.subheader("🎯 GBLUP (prédiction génomique)")
                    if st.button("Lancer GBLUP", key="btn_gblup"):
                        with st.spinner("GBLUP..."):
                            df_gebv, stats_g = run_gblup(gt_qc, matched[trait].values, ids=ind_qc["IID"].values)
                            st.session_state["gblup"] = (df_gebv, stats_g)
                    if "gblup" in st.session_state:
                        df_gebv, stats_g = st.session_state["gblup"]
                        c1, c2, c3 = st.columns(3)
                        c1.metric("h²", f"{stats_g['h2']:.3f}")
                        c2.metric("Corrélation", f"{stats_g['corr']:.3f}")
                        c3.metric("RMSE", f"{stats_g['rmse']:.3f}")
                        st.plotly_chart(plot_gblup(df_gebv), use_container_width=True)
                        st.dataframe(df_gebv.head(50), use_container_width=True)
            except Exception as e:
                st.error(f"❌ GWAS : {e}")

    # ==== SÉLECTION & FST SCAN ====
    with t_select:
        st.header("🌿 Signatures de sélection")
        st.caption("ℹ️ Pas de LD pruning — on cherche justement les pics en LD.")

        scan_type = st.radio("Analyse", ["Tajima's D (SNP)", "Tajima's D (physique)", "Fst fenêtré", "Composite Score (CSS)"], key="scan_type")

        if "Tajima" in scan_type:
            if "physique" in scan_type:
                wk = st.select_slider("Fenêtre (kb)", options=[100,250,500,1000], value=500, key="td_kb")
                if st.button("Lancer Tajima D physique", key="btn_td"):
                    with st.spinner("Calcul..."):
                        st.session_state["td"] = tajimas_scan_phys(gt_qc, snp_qc, window_kb=wk, step_kb=wk//2)
            else:
                if st.button("Lancer Tajima D (SNP)", key="btn_td_snp"):
                    with st.spinner("Calcul..."):
                        st.session_state["td"] = tajimas_scan(gt_qc, snp_qc, w=25, step=10)
            if "td" in st.session_state:
                st.plotly_chart(plot_tajima(st.session_state["td"]), use_container_width=True)
                st.dataframe(st.session_state["td"].head(100), use_container_width=True)

        elif "Fst" in scan_type:
            if st.button("Lancer Fst fenêtré", key="btn_fstw"):
                with st.spinner("Calcul..."):
                    wf = windowed_fst(gt_qc, snp_qc, labels, w=50, step=25)
                    st.session_state["wf"] = detect_outliers(wf, z_thr=3.0)
            if "wf" in st.session_state:
                wf = st.session_state["wf"]
                st.plotly_chart(plot_win_manh(wf, "MEAN_FST", "Fst fenêtré"), use_container_width=True)
                st.write(f"**Outliers (z>3)** : {wf['OUTLIER'].sum()}")
                st.dataframe(wf.head(100), use_container_width=True)

        else:
            if st.button("Calculer Composite Score", key="btn_css"):
                with st.spinner("Calcul..."):
                    st.session_state["css"] = composite_score(gt_qc, snp_qc, labels)
            if "css" in st.session_state:
                st.dataframe(st.session_state["css"].head(100), use_container_width=True)

    # ==== ANNOTATION ====
    with t_annot:
        st.header("📖 Annotation & Enrichissement")
        st.caption("ℹ️ Pas de LD pruning — annotation par SNP.")

        if gff is not None:
            st.subheader("📌 Gènes proches (GFF)")
            if "gwas_res" in st.session_state and not st.session_state["gwas_res"].empty:
                top = st.session_state["gwas_res"].head(50)
                ann = annotate_genes(top, gff, wk=50)
                if not ann.empty:
                    st.dataframe(ann, use_container_width=True)
                    if st.button("Enrichissement (top 100 gènes)", key="btn_enr"):
                        with st.spinner("Enrichr..."):
                            enr = run_enrichment(ann["GENE"].unique().tolist())
                            st.dataframe(enr.head(30), use_container_width=True)
            else:
                st.info("Lancez un GWAS pour l'annotation.")

        st.subheader("🔬 VEP (Ensembl)")
        if st.button("VEP sur top 50 SNPs GWAS", key="btn_vep"):
            if "gwas_res" in st.session_state and not st.session_state["gwas_res"].empty:
                top = st.session_state["gwas_res"].head(50)
                items = [(f"{r['CHR']}:{int(r['BP'])}", "region", f"{r['CHR']} {int(r['BP'])} . {r.get('A1','A')} {r.get('A2','G')} . . .") for _, r in top.iterrows()]
                with st.spinner("Requête VEP..."):
                    vep = run_vep_api(items, SPECIES_CONFIG[species]["vep_species"])
                    if not vep.empty:
                        vep["Traduction"] = vep["CONSEQUENCE"].apply(traduire_consequence_vep)
                        st.dataframe(vep, use_container_width=True)

        st.subheader("🕸️ Réseau STRING")
        if st.button("Requête STRING", key="btn_string"):
            if "gwas_res" in st.session_state and gff is not None:
                ann = annotate_genes(st.session_state["gwas_res"].head(50), gff, wk=50)
                genes = ann["GENE"].dropna().unique().tolist() if not ann.empty else []
                if len(genes) >= 2:
                    edges, nodes = query_string_network(genes, SPECIES_CONFIG[species]["taxon"])
                    if not edges.empty:
                        st.plotly_chart(plot_string_network(edges, nodes), use_container_width=True)

        st.subheader("🧊 Structure 3D (AlphaFold)")
        gene_in = st.text_input("Nom de gène (e.g. LALBA)", key="gene_af")
        if st.button("Charger AlphaFold", key="btn_af") and gene_in:
            with st.spinner("Recherche UniProt..."):
                uid = fetch_uniprot_id(gene_in, SPECIES_CONFIG[species]["taxon"])
                if uid:
                    pdb = fetch_alphafold_pdb(uid)
                    if pdb: render_3d_structure(pdb, style="cartoon", color_scheme="spectrum")
                    else: st.warning("Structure AlphaFold non disponible.")
                else: st.warning("Gène non trouvé dans UniProt.")

    # ==== EXPORT ====
    with t_export:
        st.header("💾 Export des données")
        st.subheader("📦 PLINK (bed/bim/fam)")
        if st.button("Générer archive PLINK", key="btn_plink"):
            data = build_plink(gt_qc, ind_qc, snp_qc, prefix="helixa")
            st.download_button("⬇️ Télécharger PLINK (.zip)", data, "helixa_plink.zip", "application/zip")

        st.subheader("📄 VCF")
        if st.button("Générer VCF", key="btn_vcf"):
            vcf_str = build_vcf(gt_qc, ind_qc, snp_qc, proj="Helixa v24")
            st.download_button("⬇️ Télécharger VCF", vcf_str.encode("utf-8"), "helixa.vcf", "text/plain")

if __name__ == "__main__":
    main()
