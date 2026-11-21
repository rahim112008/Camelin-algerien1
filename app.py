"""🧬 Universal Livestock Genomics Pipeline v23.8 (Bovin & Camelin) — stats pour publication"""
import gzip, io, warnings, zipfile, gc, hashlib, json, time, re, subprocess, tempfile, shutil, os
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
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list
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

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 1. CONFIGURATION DE LA PAGE & DESIGN HELIXA
# ============================================================
st.set_page_config(page_title="🧬 Helixa | Analyses Génomiques", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")

DEMO = str(st.query_params.get("demo", "")) == "1"
SITE_URL = "https://helixa.rahim-slimane.workers.dev/#contact"

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
    [data-testid="stSidebar"] { background-color: #080a18 !important; border-right: 1px solid rgba(255, 255, 255, 0.1) !important; }
    .stButton>button {
        background: linear-gradient(110deg, #8a2be2 0%, #00bfff 50%, #ff69b4 100%) !important;
        color: white !important; border: none !important; border-radius: 999px !important; font-weight: 600 !important; transition: 0.3s !important;
    }
    .stButton>button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 20px rgba(0, 191, 255, 0.4) !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} [data-testid="stToolbar"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- FORCER L'AFFICHAGE DU MENU SUR MOBILE ---
st.markdown("""
    <style>
    div[data-testid="stSidebarCollapsedControl"] {
        background-color: #2e7d32 !important; border-radius: 50% !important; padding: 10px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.5) !important; z-index: 9999 !important; display: block !important;
    }
    div[data-testid="stSidebarCollapsedControl"] svg { height: 30px !important; width: 30px !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURATION DES ESPÈCES
# ============================================================
SPECIES_CONFIG = {
    "Camelin (Dromadaire)": {
        "taxon": "9838", "vep_species": "camelus_dromedarius", "prefix": "camel",
        "autosomes": [str(i) for i in range(1, 37)],
        "ars": {"1": 124.7, "2": 123.8, "3": 120.7, "4": 68.3, "5": 97.5, "6": 97.5, "7": 91.2, "8": 89.4, "9": 86.7, "10": 84.1, "11": 81.5, "12": 78.9, "13": 76.3, "14": 73.8, "15": 71.2, "16": 68.6, "17": 66.1, "18": 63.5, "19": 60.9, "20": 58.4, "21": 55.8, "22": 53.2, "23": 50.7, "24": 48.1, "25": 45.5, "26": 43.0, "27": 40.4, "28": 37.8, "29": 35.3, "30": 32.7, "31": 30.1, "32": 27.6, "33": 25.0, "34": 22.4, "35": 19.9, "36": 17.3, "X": 140.0, "Y": 45.0, "MT": 16.6},
        "groupes": {
            "Production laitière": {"HAUTE_LAITIERE": ["MOJAHEEM", "MAGHATEER", "WADAH", "AWARIK"], "FAIBLE_LAITIERE": ["BIKANERI", "JAISALMERI"]},
            "Sous-espèce": {"DROMADAIRE": ["MOJAHEEM", "MAGHATEER", "WADAH", "BIKANERI"], "BACTRIANE": ["KAZAKH_BACTRIAN", "ARVANA"]}
        }
    },
    "Bovin (Taurin/Zébu)": {
        "taxon": "9913", "vep_species": "bos_taurus", "prefix": "bovine",
        "autosomes": [str(i) for i in range(1, 30)],
        "ars": {"1": 158.5, "2": 136.2, "3": 121.0, "4": 120.0, "5": 120.1, "6": 117.8, "7": 110.7, "8": 113.4, "9": 105.7, "10": 103.3, "11": 107.3, "12": 91.2, "13": 84.2, "14": 84.6, "15": 85.2, "16": 81.7, "17": 75.2, "18": 66.1, "19": 64.1, "20": 72.0, "21": 71.6, "22": 61.4, "23": 52.5, "24": 62.4, "25": 43.0, "26": 51.7, "27": 46.8, "28": 46.3, "29": 51.3, "X": 148.8, "Y": 57.8, "MT": 16.3},
        "groupes": {
            "Production": {"LAIT": ["NAR", "NBD", "PTP", "PRS"], "VIANDE": ["LJR", "SHO", "YBS"]},
            "Sous-espèce": {"ZEBU": ["PTP", "PRS"], "TAURUS": ["LJR", "SHO", "NAR", "NBD"]}
        }
    }
}

def make_group_map(group_dict):
    return {r.upper(): grp for grp, races in group_dict.items() for r in races}

def apply_group_mapping(ind_df, group_map, default="AUTRE"):
    df = ind_df.copy(); df["FID"] = df["FID"].astype(str).str.upper()
    df["FID_GROUPE"] = df["FID"].map(group_map).fillna(default)
    return df

# ============================================================
# UTILS & CACHE
# ============================================================
def _ha(x):
    if not isinstance(x, np.ndarray) or x.size == 0: return "empty"
    if x.dtype.kind in ("U", "S", "O"):
        h = hashlib.blake2b("|".join(map(str, x.ravel())).encode("utf-8", "replace"), digest_size=16)
        return f"a|{x.shape}|{h.hexdigest()}"
    return f"{x.shape}|{x.dtype}|{hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=16).hexdigest()}"

def _hn(x):
    try: return _ha(np.asarray(x))
    except Exception: return f"{type(x).__name__}|{len(x) if hasattr(x, '__len__') else 0}"

HF = {np.ndarray: _ha, pd.Series: _hn, pd.Index: _hn}
for _n in ("ArrowStringArray", "StringArray", "IntegerArray", "FloatingArray", "BooleanArray", "NumpyExtensionArray", "PandasArray", "DatetimeArray", "TimedeltaArray", "PeriodArray", "IntervalArray", "Categorical"):
    _c = getattr(pd.arrays, _n, None) or getattr(pd.core.arrays, _n, None)
    if _c: HF[_c] = _hn

def cache_data(func=None, **kw):
    def _d(f): return st.cache_data(show_spinner=False, hash_funcs=HF, **kw)(f)
    return _d if func is None else _d(func)

def impute_mean(gt):
    g = gt.astype(np.float32, copy=True); cm = np.nanmean(g, axis=0); cm = np.where(np.isnan(cm), 0.0, cm); nm = np.isnan(g)
    if nm.any(): g[nm] = np.take(cm, np.where(nm)[1])
    return g

def _std_geno(gt):
    X = impute_mean(gt); X = X - X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0
    return X / sd

def _autw(ars, autosomes):
    L = np.array([ars[c] for c in autosomes], dtype=float); return L / L.sum()

def _chk(c):
    s = str(c).upper().replace("CHROMOSOME", "").replace("CHR", "")
    if s.isdigit(): return (0, int(s), "")
    sp = {"X": 100, "Y": 101, "MT": 102, "M": 102}
    return (1, sp[s], "") if s in sp else (2, 0, s)

def _chk_num(c):
    t = _chk(c); return t[0] * 1000 + t[1]

def sort_chr_bp(df, bp="BP"):
    return (df.assign(_k=df["CHR"].map(_chk_num), _s=df["CHR"].astype(str))
              .sort_values(["_k", "_s", bp], kind="stable").drop(columns=["_k", "_s"]).reset_index(drop=True))

def _norm_chr(s):
    return s.astype(str).str.strip().str.replace(r"^(?i:chr)", "", regex=True)

def _rt(raw):
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b": return gzip.decompress(raw).decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")

def sample_fid_iid(s, fp=2):
    n = str(s).strip()
    for e in (".CEL", ".cel", ".TXT", ".txt", ".gz", ".GZ"):
        if n.endswith(e): n = n[:-len(e)]
    n = n.replace("(", "_").replace(")", "").replace("[", "_").replace("]", "_").replace(" ", "_").replace("/", "_")
    while "__" in n: n = n.replace("__", "_")
    p = n.strip("_").split("_"); return ("_".join(p[:fp]) if len(p) >= fp else p[0]), n

def _ca(a):
    a = str(a).strip() if a is not None else "0"; return "0" if a in ("", ".", "-", "NA", "nan", "None", "0", "00") else a

def _sample_blocks(n, max_snp, block=200, seed=42):
    if n <= max_snp: return np.arange(n)
    nb = int(np.ceil(n / block)); k = max(1, max_snp // block)
    ch = np.sort(np.random.default_rng(seed).choice(nb, size=min(k, nb), replace=False))
    return np.concatenate([np.arange(b * block, min((b + 1) * block, n)) for b in ch])

# ============================================================
# DIAGNOSTIC & RÉPARATION DES FICHIERS BRUTS
# ============================================================
MAX_FILE_MB = 500

def _split_line(line):
    if "\t" in line: return [x.strip() for x in line.split("\t") if x.strip()]
    if "," in line and line.count(",") > line.count(" "): return [x.strip() for x in line.split(",") if x.strip()]
    return line.split()

def diagnose_ped(raw_bytes, fn=""):
    try:
        text = _rt(raw_bytes); lines = [l for l in text.splitlines() if l.strip()]
        if not lines: return {"error": "Fichier vide"}
        sample = lines[:20]; first = _split_line(sample[0]); is_header = False
        if first and first[0].lower() in ("fid", "sample", "sample_id", "sampleid", "iid", "animal", "id", "family", "individual"): is_header = True
        data_lines = lines[1:] if is_header else lines; data_first = _split_line(data_lines[0]); is_plink = False
        if len(data_first) >= 6:
            try: int(data_first[2]); int(data_first[3]); int(data_first[4]); is_plink = True
            except (ValueError, IndexError): pass
        start = 6 if is_plink else 1; n_snp = (len(data_first) - start) // 2
        if n_snp < 1: return {"error": f"Impossible d'extraire des génotypes : {len(data_first)} colonnes"}
        sep = ("tab" if "\t" in sample[0] else "comma" if "," in sample[0] else "space")
        return {"format": "PLINK standard" if is_plink else "non-standard (IID + génotypes)", "n_individuals": len(data_lines), "n_snps": n_snp, "separator": sep, "has_header": is_header, "first_id": data_first[1] if is_plink and len(data_first) > 1 else data_first[0], "first_genotypes": data_first[start:start + 6]}
    except Exception as e: return {"error": f"{type(e).__name__}: {e}"}

def diagnose_map(raw_bytes, fn=""):
    try:
        text = _rt(raw_bytes); lines = [l for l in text.splitlines() if l.strip()]
        if not lines: return {"error": "Fichier vide"}
        sample = lines[:20]; first = _split_line(sample[0]); is_header = False
        if first and first[0].lower().lstrip("#") in ("chr", "chrom", "chromosome", "snp", "marker", "id"): is_header = True
        data_lines = lines[1:] if is_header else lines; data_first = _split_line(data_lines[0]); is_standard = False
        if len(data_first) >= 4:
            try: float(data_first[2]); int(float(data_first[3])); is_standard = True
            except (ValueError, IndexError): pass
        sep = ("tab" if "\t" in sample[0] else "comma" if "," in sample[0] else "space")
        return {"format": "PLINK standard (CHR SNP CM BP)" if is_standard else "variante", "n_lines_data": len(data_lines), "n_columns": len(data_first), "separator": sep, "has_header": is_header, "first_line_tokens": data_first[:6]}
    except Exception as e: return {"error": f"{type(e).__name__}: {e}"}

def preview_lines(raw_bytes, n=4):
    try: text = _rt(raw_bytes); return [l for l in text.splitlines() if l.strip()][:n]
    except Exception as e: return [f"❌ {e}"]

def repair_ped_streaming(raw_bytes):
    raw = gzip.decompress(raw_bytes) if (len(raw_bytes) >= 2 and raw_bytes[:2] == b"\x1f\x8b") else raw_bytes
    text = raw.decode("utf-8", errors="replace"); lines_in = text.splitlines(); del text
    def is_valid_allele(t): return t in ("A", "C", "G", "T", "a", "c", "g", "t", "0", "N")
    def find_geno_start(tokens, min_run=10):
        for i in range(len(tokens) - min_run):
            if all(is_valid_allele(t) for t in tokens[i:i + min_run]): return i
        return None
    def extract_breed(cel_name):
        clean = cel_name.replace(".CEL", "").replace(".cel", ""); clean = re.sub(r"\([^)]*\)", "", clean); parts = [p for p in clean.split("_") if p]
        if not parts: return "UNKNOWN"
        if parts[0].upper() == "DZA" and len(parts) >= 2: breed = parts[1].upper()
        else: breed = parts[0].upper()
        return re.sub(r"[^A-Z0-9]", "_", breed).strip("_") or "UNKNOWN"

    n_total = n_kept = n_fixed = n_errors = 0; breeds = {}; out_lines = []; issues = []
    for line in lines_in:
        if not line.strip(): continue
        n_total += 1; tokens = line.split()
        if not tokens: continue
        fid_raw = tokens[0]; iid_raw = tokens[1] if len(tokens) > 1 else fid_raw
        normal = (len(tokens) >= 6 and all(tokens[j] in ("0", "1", "-9", "1.0", "0.0") or tokens[j].lstrip("-").replace(".", "").isdigit() for j in range(2, 6)))
        if normal and is_valid_allele(tokens[6]) and is_valid_allele(tokens[7]): new_tokens = tokens; n_kept += 1
        else:
            start = find_geno_start(tokens[2:])
            if start is None: issues.append(f"L{n_total}: génotypes introuvables"); n_errors += 1; continue
            start += 2; genotypes = tokens[start:]; new_tokens = [fid_raw, iid_raw, "0", "0", "0", "-9"] + genotypes; n_fixed += 1
        breed = extract_breed(fid_raw); breeds[breed] = breeds.get(breed, 0) + 1
        out_lines.append("\t".join([breed, iid_raw] + new_tokens[2:]))

    cleaned = ("\n".join(out_lines) + "\n").encode("utf-8")
    lens = set(len(line.split("\t")) if "\t" in line else len(line.split()) for line in out_lines[:500])
    allele_sample = {}
    for line in out_lines[:50]:
        toks = line.split("\t") if "\t" in line else line.split()
        for a in toks[6:106]: allele_sample[a] = allele_sample.get(a, 0) + 1
    report = {"n_total": n_total, "n_kept": n_kept, "n_fixed": n_fixed, "n_errors": n_errors, "issues": issues[:10], "breeds": dict(sorted(breeds.items(), key=lambda x: -x[1])), "n_breeds": len(breeds), "line_lengths": sorted(lens), "coherent": len(lens) == 1, "alleles_sample": allele_sample, "alleles_valid": set(allele_sample.keys()).issubset({"A", "C", "G", "T", "0", "N"})}
    return cleaned, report

# ============================================================
# PARSERS
# ============================================================
def _parse_call(cell):
    s = str(cell).strip().upper()
    if not s or s in (".", "-", "0", "00", "NA", "N/A", "NAN", "NONE", "NC", "NOCALL", "--", "?"): return ("0", "0")
    for sep in ("/", "_", "|", ":"):
        if sep in s:
            parts = [x.strip() for x in s.split(sep)]
            if len(parts) == 2 and parts[0] and parts[1]: return (_ca(parts[0]), _ca(parts[1]))
            return ("0", "0")
    try:
        v = int(float(s))
        if v == 0: return ("A", "A")
        if v == 1: return ("A", "B")
        if v == 2: return ("B", "B")
        return ("0", "0")
    except (ValueError, TypeError): pass
    if len(s) == 2 and all(c in "ACGT" for c in s): return (s[0], s[1])
    if len(s) == 1 and s in "ACGT": return (s, s)
    return ("0", "0")

def parse_axiom(raw_bytes, fn=""):
    raw = gzip.decompress(raw_bytes) if (fn.endswith(".gz") or raw_bytes[:2] == b"\x1f\x8b") else raw_bytes
    text = raw.decode("utf-8", errors="replace"); lines = [l.rstrip("\r\n") for l in text.splitlines() if l.strip() and not l.startswith("#")]; del text
    if not lines: raise ValueError("Fichier vide")
    def _sp(line): return line.split("\t") if "\t" in line else line.split(",") if "," in line else line.split()
    first = _sp(lines[0]); hdr_kw = {"sample", "sample_id", "sampleid", "sample filename", "probeset_id", "probe_set_id", "probe set id", "marker", "snp", "snp_id", "id", "axiom_id", "probeset"}
    is_header = bool(first) and first[0].lower() in hdr_kw; header = _sp(lines[0]) if is_header else None
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
    ratio = sum(1 for cell in sample_row[:n_check] if looks_like_call(cell)) / max(n_check, 1)
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
            if len(calls_by_snp[i]) < n_samp: calls_by_snp[i] += ["."] * (n_samp - len(calls_by_snp[i]))
            elif len(calls_by_snp[i]) > n_samp: calls_by_snp[i] = calls_by_snp[i][:n_samp]
        gtypes = [[_parse_call(calls_by_snp[j][i]) for j in range(len(snp_ids))] for i in range(n_samp)]
        return sample_names, gtypes, len(snp_ids)
    else:
        samples, gtypes = [], []
        for line in data_lines:
            p = _sp(line)
            if len(p) < 2: continue
            samples.append(p[0].strip()); gtypes.append([_parse_call(cell) for cell in p[1:]])
        if not samples: raise ValueError("Aucun échantillon")
        c = Counter([len(r) for r in gtypes]).most_common(1)[0][0]
        for i in range(len(gtypes)):
            if len(gtypes[i]) < c: gtypes[i] += [("0","0")] * (c - len(gtypes[i]))
            elif len(gtypes[i]) > c: gtypes[i] = gtypes[i][:c]
        return samples, gtypes, c

def parse_map(b):
    text = _rt(b); rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"): continue
        p = _split_line(line)
        if not p or p[0].lower().lstrip("#") in ("chr", "chrom", "chromosome", "snp", "marker", "id"): continue
        if len(p) >= 4:
            try: cm = float(p[2]) if p[2] not in ("0", ".", "NA", "") else 0.0; bp = int(float(p[3])); rows.append({"CHR": str(p[0]), "SNP": p[1], "CM": cm, "BP": bp}); continue
            except (ValueError, IndexError): pass
        if len(p) == 3:
            try: bp = int(float(p[2])); rows.append({"CHR": str(p[0]), "SNP": p[1], "CM": 0.0, "BP": bp}); continue
            except (ValueError, IndexError): pass
    if not rows: raise ValueError(".map vide ou format non reconnu")
    df = pd.DataFrame(rows); df["CHR"] = _norm_chr(df["CHR"])
    if df["BP"].max() < 10000: df["BP"] = df["BP"] * 1_000_000
    return df

def parse_ped(b, n_map=None):
    text = _rt(b); lines = [l for l in text.splitlines() if l.strip()]
    if not lines: raise ValueError(".ped vide")
    first = _split_line(lines[0]); is_header = bool(first) and first[0].lower() in ("fid", "sample", "sample_id", "sampleid", "iid", "animal", "id", "family", "individual")
    if is_header:
        lines = lines[1:]
        if not lines: raise ValueError(".ped vide après en-tête")
    first_data = _split_line(lines[0]); is_plink = False
    if len(first_data) >= 6:
        try: int(first_data[2]); int(first_data[3]); int(first_data[4]); is_plink = True
        except (ValueError, IndexError): pass
    if is_plink: n_ped = (len(first_data) - 6) // 2; start_idx, id_idx, fid_col = 6, 1, 0
    else: n_ped = (len(first_data) - 1) // 2; start_idx, id_idx, fid_col = 1, 0, None
    if n_ped < 1: raise ValueError(f"Impossible d'extraire des génotypes : {len(first_data)} colonnes trouvées")
    n = n_ped if n_map is None else min(n_map, n_ped); ec = start_idx + 2 * n
    fids, iids, allele_to_idx, gt_rows = [], [], {}, []
    for line in lines:
        p = _split_line(line)
        if len(p) < ec: p = p + ["0"] * (ec - len(p))
        fids.append(p[fid_col] if fid_col is not None else "FAM"); iids.append(p[id_idx])
        row = np.full(2 * n, -1, dtype=np.int32)
        for k, a in enumerate(p[start_idx:start_idx + 2 * n]):
            if a not in ("0", ".", "NA", "", "N", "N/N"):
                if a not in allele_to_idx: allele_to_idx[a] = len(allele_to_idx)
                row[k] = allele_to_idx[a]
        gt_rows.append(row)
    if not iids: raise ValueError("Aucun individu valide trouvé")
    mat = np.stack(gt_rows); del gt_rows
    a1, a2 = mat[:, 0::2], mat[:, 1::2]; miss = (a1 == -1) | (a2 == -1); ok = ~miss
    K = len(allele_to_idx)
    if K == 0: raise ValueError("Aucun allèle valide trouvé dans le PED")
    cnt = np.zeros((K, n), dtype=np.int32)
    for k in range(K): cnt[k] = ((a1 == k) & ok).sum(0) + ((a2 == k) & ok).sum(0)
    idx_to_allele = {v: k for k, v in allele_to_idx.items()}; order = np.argsort(-cnt, axis=0, kind="stable")
    major = order[0]; minor = order[1] if K > 1 else order[0]
    A1 = np.array([idx_to_allele[major[j]] if cnt[major[j], j] > 0 else "." for j in range(n)])
    A2 = np.array([idx_to_allele[minor[j]] if cnt[minor[j], j] > 0 else "." for j in range(n)])
    gt = ((a1 == minor[None, :]).astype(np.float32) + (a2 == minor[None, :]).astype(np.float32)); gt[miss] = np.nan
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
                for k, s in enumerate(p[9:9 + len(samples)]):
                    try:
                        al = s.split(":")[gi].replace("|", "/").split("/")
                        if not any(a in (".", "") for a in al): gts[k] = sum(int(a) for a in al)
                    except (ValueError, IndexError): pass
                try: bp = int(float(p[1]))
                except (ValueError, TypeError): continue
                rows.append({"CHR": str(p[0]), "SNP": p[2] if p[2] not in ("", ".") else f"{p[0]}:{p[1]}", "CM": 0.0, "BP": bp, "A1": p[3], "A2": p[4]})
                gts_list.append(gts)
    if not rows: raise ValueError("Aucun variant VCF")
    gt = np.column_stack(gts_list).astype(np.float32)
    for j in range(gt.shape[1]):
        v = gt[:, j][~np.isnan(gt[:, j])]
        if len(v) > 0 and v.mean() / 2.0 > 0.5: gt[:, j] = 2.0 - gt[:, j]; rows[j]["A1"], rows[j]["A2"] = rows[j]["A2"], rows[j]["A1"]
    df_snp = pd.DataFrame(rows); df_snp["CHR"] = _norm_chr(df_snp["CHR"])
    return gt, pd.DataFrame({"FID": samples, "IID": samples}), df_snp

def _gff_attr(a, key):
    for part in a.split(";"):
        part = part.strip()
        if part.startswith(key + "="): return part[len(key) + 1:]
    return None

def parse_gff3(b):
    genes = []
    for line in _rt(b).splitlines():
        if line.startswith("#") or not line.strip(): continue
        p = line.split("\t")
        if len(p) < 9 or p[2] not in ("gene", "ncRNA_gene", "pseudogene"): continue
        a = p[8]; name = None
        if 'gene_name "' in a: name = a.split('gene_name "')[1].split('"')[0]
        elif 'gene_id "' in a: name = a.split('gene_id "')[1].split('"')[0]
        else:
            for k in ("Name", "gene", "gene_name", "ID"):
                name = _gff_attr(a, k)
                if name: break
        if not name: name = "Unknown"
        for pre in ("gene:", "gene-"):
            if name.startswith(pre): name = name[len(pre):]
        chrom = p[0][3:] if p[0].lower().startswith("chr") else p[0]
        try: genes.append({"CHR": str(chrom), "START": int(p[3]), "END": int(p[4]), "GENE": name})
        except ValueError: continue
    df = pd.DataFrame(genes, columns=["CHR", "START", "END", "GENE"])
    return df.drop_duplicates().reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_gff(b): return parse_gff3(b)

def parse_gmt(file_bytes):
    text = file_bytes.decode("utf-8", errors="replace"); sets = {}
    for line in text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3: continue
        genes = [g for g in parts[2:] if g]
        if genes: sets[parts[0]] = genes
    return sets

def load_pheno(b, fn=""):
    raw = gzip.decompress(b) if fn.endswith(".gz") else b
    text = raw.decode("utf-8", errors="replace"); sep = "\t" if text.count("\t") > text.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python")

def match_pheno(ind_df, pheno_df, mapping_df=None):
    ind = ind_df.copy(); ind["_k"] = ind["IID"].astype(str)
    if mapping_df is not None:
        p, m = pheno_df.copy(), mapping_df.copy()
        id_c = next((c for c in ["Animal", "IID", "id", "ID"] if c in p.columns), None)
        if id_c is None: raise ValueError("Colonne ID absente.")
        if id_c != "IID": p = p.drop(columns=["IID"], errors="ignore")
        p[id_c] = p[id_c].astype(str); map_a = next((c for c in m.columns if c != "IID"), None)
        if map_a is None or "IID" not in m.columns: raise ValueError("Le mapping doit contenir IID + une colonne d'identifiant.")
        m[map_a] = m[map_a].astype(str); m["IID"] = m["IID"].astype(str)
        pm = p.merge(m, left_on=id_c, right_on=map_a, how="left"); pm["_k"] = pm["IID"].astype(str); pm = pm.drop_duplicates("_k")
        out = ind[["_k", "FID", "IID"]].merge(pm.drop(columns=["IID", "FID"], errors="ignore"), on="_k", how="left")
        return out.drop(columns="_k"), id_c
    id_c = next((c for c in ["IID", "id", "ID", "sample", "Animal", "animal"] if c in pheno_df.columns), None)
    if id_c is None: raise ValueError("Aucune colonne ID.")
    p = pheno_df.copy(); p["_k"] = p[id_c].astype(str); p = p.drop_duplicates("_k")
    out = ind[["_k", "FID", "IID"]].merge(p.drop(columns=[id_c, "FID"], errors="ignore"), on="_k", how="left")
    return out.drop(columns="_k"), id_c

def gen_demo(n_ind=150, n_snp=800, n_pop=4, seed=42, autosomes=None, ars=None):
    rng = np.random.default_rng(seed)
    if not autosomes or not ars: autosomes = [str(i) for i in range(1, 30)]; ars = {c: 100.0 for c in autosomes}
    pops = [f"POP_{i}" for i in range(1, min(n_pop, 10) + 1)]
    pa = rng.beta(0.6, 0.6, n_snp); gt = np.zeros((n_ind, n_snp), dtype=np.float32); ir = []; k = 0
    for pi_, p in enumerate(pops):
        pk = np.clip(pa + rng.normal(0, 0.15, n_snp), 0.02, 0.98)
        for i in range(n_ind // len(pops) + (1 if pi_ < n_ind % len(pops) else 0)):
            gt[k] = (rng.random(n_snp) < pk).astype(np.int8) + (rng.random(n_snp) < pk).astype(np.int8)
            ir.append({"FID": p, "IID": f"{p}_{i + 1:03d}"}); k += 1
    gt[rng.random(gt.shape) < 0.02] = np.nan
    ch = rng.choice(autosomes, n_snp, p=_autw(ars, autosomes)); bp = np.array([rng.integers(1, int(ars[c] * 1_000_000)) for c in ch], dtype=np.int64)
    snp = pd.DataFrame({"CHR": ch.astype(str), "SNP": [f"rs{i:07d}" for i in range(n_snp)], "CM": bp / 1e6, "BP": bp, "A1": "A", "A2": "G"})
    order = np.lexsort((bp, snp["CHR"].map(_chk_num).values))
    return gt[:, order], pd.DataFrame(ir), snp.iloc[order].reset_index(drop=True)

def rec_fid(ind):
    df = ind.copy()
    if df["FID"].nunique() <= 1: df["FID"] = df["IID"].astype(str).str.split("_").str[0]
    return df

def clean_ped_to_plink(raw_bytes, n_max_snp=None):
    gt, ind, a1, a2 = parse_ped(raw_bytes, n_max_snp); n, m = gt.shape; lines = []
    for i in range(n):
        geno = []
        for j in range(m):
            v = gt[i, j]
            if np.isnan(v): geno += ["0", "0"]
            elif v == 0: geno += [a1[j], a1[j]]
            elif v == 1: geno += [a1[j], a2[j]]
            else: geno += [a2[j], a2[j]]
        lines.append("\t".join([str(ind.iloc[i]["FID"]), str(ind.iloc[i]["IID"]), "0", "0", "0", "-9"] + geno))
    return "\n".join(lines) + "\n"

# ============================================================
# QC
# ============================================================
def miss_ind(g): return np.isnan(g).mean(axis=1)
def miss_snp(g): return np.isnan(g).mean(axis=0)
def maf(g): p = np.nanmean(g, axis=0) / 2.0; return np.minimum(p, 1.0 - p)

def het(g):
    ok = ~np.isnan(g); n = ok.sum(1)
    return np.where(n > 0, ((g == 1) & ok).sum(1) / np.maximum(n, 1), np.nan)

def hwe_exact(nh, n0, n2):
    n = nh + n0 + n2
    if n == 0 or (nh == 0 and (n0 == 0 or n2 == 0)): return 1.0
    rare = 2 * min(n0, n2) + nh; mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2: mid += 1
    pr = np.zeros(rare + 1); pr[mid] = 1.0; s = 1.0
    ch, cr, cc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch <= rare - 2: pr[ch + 2] = pr[ch] * 4 * cr * cc / ((ch + 2) * (ch + 1)); s += pr[ch + 2]; ch += 2; cr -= 1; cc -= 1
    ch, cr, cc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch >= 2: pr[ch - 2] = pr[ch] * ch * (ch - 1) / (4 * (cr + 1) * (cc + 1)); s += pr[ch - 2]; ch -= 2; cr += 1; cc += 1
    po = pr[nh] if nh < len(pr) else 0.0
    return float(min(pr[pr <= po + 1e-7].sum() / s, 1.0))

@cache_data
def hwe_pvals(g):
    pv = np.full(g.shape[1], np.nan)
    n0 = ((g == 0) & ~np.isnan(g)).sum(0); n1 = ((g == 1) & ~np.isnan(g)).sum(0); n2 = ((g == 2) & ~np.isnan(g)).sum(0); nv = n0 + n1 + n2
    if g.shape[1] <= 20000:
        for j in range(g.shape[1]):
            if nv[j] >= 5: pv[j] = hwe_exact(int(n1[j]), int(n0[j]), int(n2[j]))
    else:
        p = (n1 + 2 * n2) / (2.0 * np.where(nv > 0, nv, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            eh = 2.0 * p * (1.0 - p) * nv; ok = (eh >= 5) & np.isfinite(eh); chi2 = (np.abs(n1 - eh) - 0.5) ** 2 / eh; pv[ok] = stats.chi2.sf(chi2[ok], df=1)
    return pv

def sync_dims(gt, ind_df, snp_df):
    n = min(gt.shape[0], len(ind_df)); m = min(gt.shape[1], len(snp_df))
    return gt[:n, :m], ind_df.iloc[:n].reset_index(drop=True), snp_df.iloc[:m].reset_index(drop=True)

def apply_qc(gt, ind_df, snp_df, params, autosomes, keep_autosomes=True, rem_out=True):
    gt, ind_df, snp_df = sync_dims(gt, ind_df, snp_df); n0, m0 = gt.shape; tr = {}
    if keep_autosomes:
        auto = {str(a).upper() for a in autosomes}
        mask = snp_df["CHR"].astype(str).str.upper().str.replace(r"^CHR", "", regex=True).isin(auto).values
        gt = gt[:, mask]; snp_df = snp_df[mask].reset_index(drop=True); tr["autosomes"] = int(m0 - gt.shape[1])
    m = gt.shape[1]
    k = miss_snp(gt) <= params["geno"]; gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True); tr["geno"] = int(m - gt.shape[1]); m = gt.shape[1]
    k = miss_ind(gt) <= params["mind"]; gt = gt[k]; ind_df = ind_df[k].reset_index(drop=True); tr["mind"] = int(n0 - gt.shape[0])
    mf = maf(gt); k = np.isfinite(mf) & (mf >= params["maf"]); gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True); tr["maf"] = int(m - gt.shape[1]); m = gt.shape[1]
    if gt.shape[1] == 0: raise ValueError("Tous les SNPs exclus (MAF)")
    pv = hwe_pvals(gt); k = np.isnan(pv) | (pv >= params["hwe"])
    gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True); tr["hwe"] = int(m - gt.shape[1])
    if gt.shape[1] == 0: raise ValueError("Tous les SNPs exclus (HWE)")
    n_pre = int(gt.shape[0])
    if rem_out:
        h = het(gt); kp = np.ones(len(h), dtype=bool)
        for f in ind_df["FID"].unique():
            msk = (ind_df["FID"] == f).values; sub = h[msk]
            if len(sub) >= 5 and np.nanstd(sub) > 1e-9:
                z = (sub - np.nanmean(sub)) / np.nanstd(sub); idx = np.where(msk)[0]; kp[idx[np.abs(z) > params["het_sd"]]] = False
        gt = gt[kp]; ind_df = ind_df[kp].reset_index(drop=True); tr["het"] = int(n_pre - len(ind_df))
    gc.collect()
    return gt, ind_df, snp_df, {"n_ind_init": int(n0), "n_snp_init": int(m0), "n_ind_pre": n_pre, "n_ind_final": int(gt.shape[0]), "n_snp_final": int(gt.shape[1]), "excluded_ind": int(n0 - gt.shape[0]), "excluded_snp": int(m0 - gt.shape[1]), "trace": tr}

# ============================================================
# ALGORITHMES
# ============================================================
def _wc_abc(gt, labels, min_n=2):
    labels = np.asarray(labels).astype(str); pops = np.unique(labels); m = gt.shape[1]
    n = np.zeros((len(pops), m)); p = np.zeros_like(n); h = np.zeros_like(n)
    for k, pop in enumerate(pops):
        g = gt[labels == pop]; ok = ~np.isnan(g); nn = ok.sum(0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            pk = np.where(nn > 0, np.nansum(g, 0) / (2 * nn), 0.0); hk = np.where(nn > 0, ((g == 1) & ok).sum(0) / nn, 0.0)
        use = nn >= min_n
        n[k] = np.where(use, nn, 0.0); p[k] = np.where(use, pk, 0.0); h[k] = np.where(use, hk, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (n > 0).sum(0).astype(float); nsum = n.sum(0); nbar = nsum / r; pbar = (n * p).sum(0) / nsum
        s2 = (n * (p - pbar) ** 2).sum(0) / (nbar * (r - 1)); hbar = (n * h).sum(0) / nsum
        nc = (r * nbar - (n ** 2).sum(0) / (r * nbar)) / (r - 1)
        a = (nbar / nc) * (s2 - (1.0 / (nbar - 1)) * (pbar * (1 - pbar) - (r - 1) / r * s2 - hbar / 4))
        b = (nbar / (nbar - 1)) * (pbar * (1 - pbar) - (r - 1) / r * s2 - (2 * nbar - 1) / (4 * nbar) * hbar)
        den = a + b + hbar / 2.0
    valid = (r >= 2) & (nbar > 1) & (nc > 0) & np.isfinite(den) & np.isfinite(a)
    return np.where(valid, a, np.nan), np.where(valid, b, np.nan), np.where(valid, hbar / 2.0, np.nan)

def _wc_components(gt, labels, min_n=2):
    a, b, c = _wc_abc(gt, labels, min_n); return a, a + b + c

@cache_data
def fst_snp_wc(gt, labels):
    a, den = _wc_components(gt, labels)
    with np.errstate(invalid="ignore", divide="ignore"): f = a / den
    return np.where(np.isfinite(f) & (den > 1e-9), np.clip(f, 0.0, None), np.nan)

def fst_global_wc(gt, labels):
    a, den = _wc_components(gt, labels); ok = np.isfinite(a) & np.isfinite(den)
    return float(max(0.0, a[ok].sum() / den[ok].sum())) if ok.any() and den[ok].sum() > 0 else np.nan

@cache_data
def pca_analysis(gt, nc=10):
    X = impute_mean(gt); X = X - X.mean(0); n = min(nc, X.shape[0] - 1, X.shape[1]); p = PCA(n_components=n)
    return p.fit_transform(X), p.explained_variance_ratio_ * 100.0

@cache_data
def mds_analysis(gt, nc=5):
    X = impute_mean(gt); n = X.shape[0]; D = np.zeros((n, n), dtype=np.float32)
    for i in range(n): D[i] = np.abs(X - X[i]).sum(1) / X.shape[1]
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
            freq = np.where(n > 0, np.nansum(sub, 0) / (2 * n), np.nan); ho_snp = np.where(n > 0, ((sub == 1) & ok).sum(0) / n, np.nan)
        v = np.isfinite(freq)
        if not v.any(): continue
        f = freq[v]; q = 1 - f; he = float(np.mean(2 * f * q)); ho = float(np.nanmean(ho_snp[v]))
        pic = float(np.mean(1 - (f ** 2 + q ** 2) - 2 * (f ** 2) * (q ** 2)))
        rows.append({"Population": p, "N": int(sub.shape[0]), "H_Observee (Ho)": round(ho, 4), "H_Attendue (He)": round(he, 4), "Fis (Consanguinité)": round(1 - ho / he, 4) if he > 0 else 0.0, "PIC": round(pic, 4), "MAF_moyenne": round(float(np.mean(np.minimum(f, q))), 4), "SNPs_Polymorphes": int(((f > 0) & (f < 1)).sum())})
    return pd.DataFrame(rows)

@cache_data
def king_mat(gt):
    ok = ~np.isnan(gt); H = ((gt == 1) & ok).astype(np.float32); G0 = ((gt == 0) & ok).astype(np.float32); G2 = ((gt == 2) & ok).astype(np.float32)
    h = H.sum(1).astype(np.float64); BH = H @ H.T; ibs0 = G0 @ G2.T + G2 @ G0.T; d = h[:, None] + h[None, :]
    with np.errstate(divide="ignore", invalid="ignore"): K = (BH - 2 * ibs0) / d
    K[~np.isfinite(K)] = 0.0; np.fill_diagonal(K, 0.5); return K

@cache_data
def grm_vanraden(gt):
    X = impute_mean(gt); p = X.mean(0) / 2.0; Z = X - 2 * p; den = 2 * np.sum(p * (1 - p))
    if den <= 0: return np.eye(gt.shape[0])
    return ((Z @ Z.T) / den).astype(np.float64)

@cache_data
def admixture_nmf(gt, K=3, seed=42):
    m = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=500); W = m.fit_transform(np.clip(impute_mean(gt), 0.0, 2.0)); s = W.sum(1, keepdims=True); s[s == 0] = 1.0
    return W / s, m.components_

@cache_data
def choose_best_k(gt, k_min=2, k_max=8, seed=42):
    X = np.clip(impute_mean(gt), 0.0, 2.0); rows = []
    for k in range(k_min, k_max + 1):
        m = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=400); W = m.fit_transform(X)
        rows.append({"K": k, "RMSE": float(np.sqrt(np.mean((X - W @ m.components_) ** 2)))})
    return pd.DataFrame(rows)

@cache_data
def ld_pruning(gt, chrs, w=50, step=5, r2=0.2):
    n = gt.shape[1]; keep = np.ones(n, dtype=bool)
    if n < 3: return keep
    chrs = np.asarray(chrs).astype(str); X = _std_geno(gt).astype(np.float32); N = X.shape[0]
    for c in pd.unique(chrs):
        idx = np.where(chrs == c)[0]
        for s in range(0, len(idx), step):
            win = idx[s:s + w]; win = win[keep[win]]
            if len(win) < 2: continue
            R2 = ((X[:, win].T @ X[:, win]) / N) ** 2
            for a in range(len(win) - 1):
                if not keep[win[a]]: continue
                hits = np.where(R2[a, a + 1:] > r2)[0] + a + 1; keep[win[hits]] = False
    return keep

def _pair_r2(X, bp, chrs, win=100, max_bp=None):
    n = X.shape[1]; N = X.shape[0]; D, R = [], []
    for i in range(n - 1):
        j1 = min(i + 1 + win, n); same = chrs[i + 1:j1] == chrs[i]
        if not same.any(): continue
        d = np.abs(bp[i + 1:j1] - bp[i]); ok = same & (d > 0)
        if max_bp is not None: ok &= (d <= max_bp)
        if ok.any(): r = (X[:, i + 1:j1][:, ok].T @ X[:, i]) / N; D.append(d[ok]); R.append(r.astype(np.float64) ** 2)
    return (np.concatenate(D), np.concatenate(R)) if D else (np.array([]), np.array([]))

def _sorted_snps(gt, snp_df):
    s = snp_df.reset_index(drop=True).copy(); s["CHR"] = s["CHR"].astype(str); s["BP"] = s["BP"].astype(float)
    order = np.lexsort((s["BP"].values, s["CHR"].map(_chk_num).values))
    return gt[:, order], s.iloc[order].reset_index(drop=True)

@cache_data
def ld_decay(gt, snp_df, max_kb=1000, max_snp=20000):
    if gt.shape[1] < 10: return pd.DataFrame(columns=["dist_kb", "r2", "count"])
    gt, s = _sorted_snps(gt, snp_df)
    idx = _sample_blocks(gt.shape[1], max_snp); gt = gt[:, idx]; s = s.iloc[idx].reset_index(drop=True)
    X = _std_geno(gt); dist, r2 = _pair_r2(X, s["BP"].values, s["CHR"].values, 100, max_kb * 1000)
    if len(dist) == 0: return pd.DataFrame(columns=["dist_kb", "r2", "count"])
    df = pd.DataFrame({"dist_kb": dist / 1000.0, "r2": r2}); df["bin"] = pd.cut(df["dist_kb"], bins=np.linspace(0, max_kb, 30), labels=False, include_lowest=True)
    return df.groupby("bin").agg(dist_kb=("dist_kb", "mean"), r2=("r2", "mean"), count=("r2", "count")).dropna().sort_values("dist_kb").reset_index(drop=True)

@cache_data
def ne_historical(gt, snp_df, max_snp=10000):
    N = gt.shape[0]
    if N < 5 or gt.shape[1] < 20: return pd.DataFrame({"GenAgo": [], "Ne": []})
    gt, s = _sorted_snps(gt, snp_df); idx = _sample_blocks(gt.shape[1], max_snp); gt = gt[:, idx]; s = s.iloc[idx].reset_index(drop=True)
    X = _std_geno(gt); d, r2 = _pair_r2(X, s["BP"].values, s["CHR"].values, 50, 50e6); d = d / 1e6
    m = (d >= 0.001) & (d <= 50)
    if m.sum() < 20: return pd.DataFrame({"GenAgo": [], "Ne": []})
    d, r2 = d[m], r2[m]; bins = np.logspace(np.log10(d.min()), np.log10(d.max()), 20); rows = []
    for k in range(len(bins) - 1):
        sel = (d >= bins[k]) & (d < bins[k + 1])
        if sel.sum() >= 10: rows.append((np.sqrt(bins[k] * bins[k + 1]), r2[sel].mean()))
    if len(rows) < 3: return pd.DataFrame({"GenAgo": [], "Ne": []})
    dm, rm = np.array(rows).T; c = dm * 0.01; r2a = np.clip(rm - 1.0 / N, 1e-6, 0.999); ne = (1.0 / r2a - 1.0) / (4.0 * c); t = 1.0 / (2.0 * c)
    df = pd.DataFrame({"GenAgo": t, "Ne": ne}).replace([np.inf, -np.inf], np.nan).dropna(); df = df[df["Ne"] > 0].sort_values("GenAgo").reset_index(drop=True)
    if len(df) > 3: df["Ne"] = df["Ne"].rolling(3, center=True, min_periods=1).mean()
    return df

@cache_data
def detect_roh(gt, chrs, bps, min_snps=30, min_kb=500.0):
    chrs = np.asarray(chrs).astype(str); bps = np.asarray(bps).astype(np.int64); n_ind = gt.shape[0]; rohs = []; fb = np.zeros(n_ind); tot = 0.0
    for c in pd.unique(chrs):
        idxs = np.where(chrs == c)[0]
        if len(idxs) < min_snps: continue
        idxs = idxs[np.argsort(bps[idxs], kind="stable")]; pos = bps[idxs]; sp = pos[-1] - pos[0]
        if sp < 1: continue
        tot += sp; G = gt[:, idxs]
        for i in range(n_ind):
            bnd = np.concatenate(([-1], np.where(G[i] == 1)[0], [len(idxs)]))
            for a, b in zip(bnd[:-1], bnd[1:]):
                s_, e_ = a + 1, b - 1
                if e_ < s_: continue
                nr = e_ - s_ + 1; lkb = (pos[e_] - pos[s_]) / 1000.0
                if nr >= min_snps and lkb >= min_kb: rohs.append({"IID_idx": i, "CHR": c, "start_bp": int(pos[s_]), "end_bp": int(pos[e_]), "n_snp": int(nr), "length_kb": float(lkb)}); fb[i] += pos[e_] - pos[s_]
    cols = ["IID_idx", "CHR", "start_bp", "end_bp", "n_snp", "length_kb"]
    return (pd.DataFrame(rohs, columns=cols)), (fb / tot if tot > 0 else fb)

@cache_data
def fst_pairwise(gt, labels, use_wc84=True):
    labels = np.asarray(labels).astype(str); pops = sorted(np.unique(labels)); K = len(pops); M = np.full((K, K), np.nan); np.fill_diagonal(M, 0.0)
    for i in range(K):
        for j in range(i + 1, K):
            mi, mj = labels == pops[i], labels == pops[j]
            if use_wc84: v = fst_global_wc(gt[mi | mj], labels[mi | mj])
            else:
                g1, g2 = gt[mi], gt[mj]; p1 = np.nanmean(g1, 0) / 2.0; p2 = np.nanmean(g2, 0) / 2.0; n1 = (~np.isnan(g1)).sum(0); n2 = (~np.isnan(g2)).sum(0)
                dn = np.where(n1 + n2 > 0, n1 + n2, np.nan); pb = (p1 * n1 + p2 * n2) / dn; hs = (2 * p1 * (1 - p1) * n1 + 2 * p2 * (1 - p2) * n2) / dn; ht = 2 * pb * (1 - pb)
                ok = np.isfinite(ht) & np.isfinite(hs); v = float(max(0.0, (ht[ok] - hs[ok]).sum() / ht[ok].sum())) if ok.any() and ht[ok].sum() > 0 else np.nan
            M[i, j] = M[j, i] = v
    return M, pops

@cache_data
def windowed_fst(gt, snp_df, labels, w=50, step=25):
    a, den = _wc_components(gt, labels); snp = snp_df.reset_index(drop=True).copy(); n = min(len(snp), len(a)); snp = snp.iloc[:n].copy(); a = a[:n]; den = den[:n]
    snp["_a"] = a; snp["_d"] = den
    with np.errstate(invalid="ignore", divide="ignore"): snp["FST"] = np.where(np.isfinite(den) & (den > 1e-9), np.clip(a / den, 0.0, None), np.nan)
    res = []
    for c in snp["CHR"].astype(str).unique():
        sub = snp[snp["CHR"].astype(str) == c].sort_values("BP").reset_index(drop=True)
        if len(sub) < w: continue
        A, Dn, F, B, S = sub["_a"].values, sub["_d"].values, sub["FST"].values, sub["BP"].values, sub["SNP"].values
        for i in range(0, len(sub) - w + 1, step):
            sl = slice(i, i + w); okm = np.isfinite(A[sl]) & np.isfinite(Dn[sl])
            if okm.sum() < 5: continue
            dd = Dn[sl][okm].sum()
            if dd <= 1e-9: continue
            f_sl = F[sl]; top = int(np.nanargmax(f_sl)) if np.isfinite(f_sl).any() else 0
            res.append({"CHR": c, "START": int(B[sl].min()), "END": int(B[sl].max()), "MID": int((B[sl].min() + B[sl].max()) / 2), "N_SNP": w, "MEAN_FST": float(max(0.0, A[sl][okm].sum() / dd)), "MAX_FST": float(np.nanmax(f_sl)) if np.isfinite(f_sl).any() else np.nan, "TOP_SNP": S[sl][top]})
    return pd.DataFrame(res, columns=["CHR", "START", "END", "MID", "N_SNP", "MEAN_FST", "MAX_FST", "TOP_SNP"])

@cache_data
def detect_outliers(win_df, z_thr=3.0):
    df = win_df.copy()
    if df.empty: df["Z_FST"] = []; df["OUTLIER"] = []; return df
    mu, sd = df["MEAN_FST"].mean(), df["MEAN_FST"].std(); df["Z_FST"] = (df["MEAN_FST"] - mu) / sd if sd > 1e-9 else 0.0; df["OUTLIER"] = df["Z_FST"] >= z_thr
    return df.sort_values("MEAN_FST", ascending=False).reset_index(drop=True)

@cache_data
def tajimas_window(genos):
    ni, ns = genos.shape
    if ns < 3 or ni < 4: return np.nan, 0, np.nan, np.nan
    S = 0; piv = []; thv = []
    for j in range(ns):
        col = genos[:, j]; col = col[~np.isnan(col)]
        if len(col) < 2: continue
        p = col.mean() / 2.0; N = 2 * len(col)
        if 0 < p < 1: S += 1; piv.append(2 * p * (1 - p) * N / (N - 1)); thv.append(1.0 / np.sum(1.0 / np.arange(1, N)))
    if S == 0 or not piv: return np.nan, int(S), np.nan, np.nan
    pi = float(np.sum(piv)); th = float(np.sum(thv)); n = ni * 2; a1 = np.sum(1.0 / np.arange(1, n)); a2 = np.sum(1.0 / (np.arange(1, n) ** 2)); b1 = (n + 1) / (3 * (n - 1)); b2 = 2 * (n ** 2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - 1 / a1; c2 = b2 - (n + 2) / (a1 * n) + a2 / (a1 ** 2); e1 = c1 / a1; e2 = c2 / (a1 ** 2 + a2); den = np.sqrt(e1 * S + e2 * S * (S - 1))
    if den <= 1e-12: return np.nan, int(S), pi, th
    return float((pi - S / a1) / den), int(S), pi, th

@cache_data
def tajimas_scan(gt, snp_df, w=25, step=10):
    snp = snp_df.copy().reset_index(drop=True); n = min(gt.shape[1], len(snp)); snp = snp.iloc[:n].reset_index(drop=True); gt = gt[:, :n]; res = []
    for c in snp["CHR"].unique():
        idx = np.where(snp["CHR"].values == c)[0]
        if len(idx) < w: continue
        idx = idx[np.argsort(snp["BP"].values[idx])]
        for i in range(0, len(idx) - w + 1, step):
            wi = idx[i:i + w]; D, S, pis, ths = tajimas_window(gt[:, wi]); bpw = snp["BP"].values[wi]; span_kb = max((bpw.max() - bpw.min()) / 1000.0, 1e-9)
            if not np.isnan(D): res.append({"CHR": str(c), "START": int(bpw.min()), "END": int(bpw.max()), "MID": int((bpw.min() + bpw.max()) / 2), "N_SNP": len(wi), "S": S, "TAJIMA_D": D, "PI_PAR_SNP": pis / len(wi), "THETA_W_PAR_SNP": ths / len(wi), "PI_PAR_KB": pis / span_kb})
    return pd.DataFrame(res, columns=["CHR", "START", "END", "MID", "N_SNP", "S", "TAJIMA_D", "PI_PAR_SNP", "THETA_W_PAR_SNP", "PI_PAR_KB"])

@cache_data
def tajimas_scan_phys(gt, snp_df, window_kb=500, step_kb=250, min_snps=10):
    """Tajima's D par fenêtres physiques (bp)"""
    snp = snp_df.copy().reset_index(drop=True)
    n = min(gt.shape[1], len(snp))
    snp = snp.iloc[:n].reset_index(drop=True); gt = gt[:, :n]
    res = []
    step_bp = int(step_kb * 1000); win_bp = int(window_kb * 1000)
    for c in snp["CHR"].unique():
        idx_c = np.where(snp["CHR"].values == c)[0]
        if len(idx_c) < min_snps: continue
        idx_c = idx_c[np.argsort(snp["BP"].values[idx_c])]
        bp_c = snp["BP"].values[idx_c]
        for start in range(int(bp_c[0]), int(bp_c[-1]) + 1, step_bp):
            end = start + win_bp
            mask = (bp_c >= start) & (bp_c < end)
            if mask.sum() < min_snps: continue
            wi = idx_c[mask]
            D, S, pis, ths = tajimas_window(gt[:, wi])
            bpw = bp_c[mask]; span_kb = max((bpw.max() - bpw.min()) / 1000.0, 1e-9)
            if not np.isnan(D):
                res.append({"CHR": str(c), "START": int(start), "END": int(end),
                            "MID": int((start + end) // 2), "N_SNP": int(len(wi)), "S": S,
                            "TAJIMA_D": D, "PI_PAR_SNP": pis/len(wi), "THETA_W_PAR_SNP": ths/len(wi), "PI_PAR_KB": pis/span_kb})
    return pd.DataFrame(res, columns=["CHR", "START", "END", "MID", "N_SNP", "S", "TAJIMA_D", "PI_PAR_SNP", "THETA_W_PAR_SNP", "PI_PAR_KB"])

@cache_data
def shannon_window(genos):
    ni, ns = genos.shape
    if ns == 0 or ni < 2: return np.nan
    h_vals = []
    for j in range(ns):
        col = genos[:, j]
        col = col[~np.isnan(col)]
        if len(col) < 2: continue
        p = col.mean() / 2.0
        if 0 < p < 1:
            q = 1.0 - p
            h = -(p * np.log(p) + q * np.log(q))
            h_vals.append(h)
    return float(np.mean(h_vals)) if h_vals else np.nan

@cache_data
def shannon_scan(gt, snp_df, w=25, step=10):
    snp = snp_df.copy().reset_index(drop=True); n = min(gt.shape[1], len(snp))
    snp = snp.iloc[:n].reset_index(drop=True); gt = gt[:, :n]; res = []
    for c in snp["CHR"].unique():
        idx = np.where(snp["CHR"].values == c)[0]
        if len(idx) < w: continue
        idx = idx[np.argsort(snp["BP"].values[idx])]
        for i in range(0, len(idx) - w + 1, step):
            wi = idx[i:i+w]; H = shannon_window(gt[:, wi]); bpw = snp["BP"].values[wi]
            if not np.isnan(H):
                res.append({"CHR": str(c), "START": int(bpw.min()), "END": int(bpw.max()),
                            "MID": int((bpw.min() + bpw.max()) / 2), "N_SNP": len(wi), "SHANNON_H": H})
    return pd.DataFrame(res, columns=["CHR", "START", "END", "MID", "N_SNP", "SHANNON_H"])

@cache_data
def shannon_scan_phys(gt, snp_df, window_kb=500, step_kb=250, min_snps=10):
    snp = snp_df.copy().reset_index(drop=True)
    n = min(gt.shape[1], len(snp))
    snp = snp.iloc[:n].reset_index(drop=True); gt = gt[:, :n]; res = []
    step_bp = int(step_kb * 1000); win_bp = int(window_kb * 1000)
    for c in snp["CHR"].unique():
        idx_c = np.where(snp["CHR"].values == c)[0]
        if len(idx_c) < min_snps: continue
        idx_c = idx_c[np.argsort(snp["BP"].values[idx_c])]
        bp_c = snp["BP"].values[idx_c]
        for start in range(int(bp_c[0]), int(bp_c[-1]) + 1, step_bp):
            end = start + win_bp
            mask = (bp_c >= start) & (bp_c < end)
            if mask.sum() < min_snps: continue
            wi = idx_c[mask]
            H = shannon_window(gt[:, wi])
            if not np.isnan(H):
                res.append({"CHR": str(c), "START": int(start), "END": int(end),
                            "MID": int((start + end) // 2), "N_SNP": int(len(wi)), "SHANNON_H": float(H)})
    return pd.DataFrame(res, columns=["CHR", "START", "END", "MID", "N_SNP", "SHANNON_H"])

@cache_data
def composite_score(gt, snp_df, labels):
    fst = fst_snp_wc(gt, labels); ok = ~np.isnan(gt); n = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        hom = np.where(n > 0, (((gt == 0) | (gt == 2)) & ok).sum(0) / n, np.nan); mv = np.nanmean(gt, axis=0) / 2.0
    mv = np.minimum(mv, 1 - mv); df = snp_df.copy().reset_index(drop=True); m = min(len(df), len(fst)); df = df.iloc[:m].reset_index(drop=True)
    df["FST"] = fst[:m]; df["HOM"] = hom[:m]; df["MAF"] = mv[:m]
    def norm(x): mu, sd = np.nanmean(x), np.nanstd(x); return (x - mu) / sd if sd > 1e-9 else np.zeros_like(x)
    df["Z_FST"] = norm(df["FST"].values); df["Z_HOM"] = norm(df["HOM"].values); df["CSS"] = np.abs(df["Z_FST"] * df["Z_HOM"])
    return df.sort_values("CSS", ascending=False).reset_index(drop=True)

def _reynolds_D(g, labels, pops):
    ph = {p: np.nanmean(g[labels == p], 0) / 2.0 for p in pops}; K = len(pops); D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            pi_, pj = ph[pops[i]], ph[pops[j]]; ok = np.isfinite(pi_) & np.isfinite(pj)
            num = np.sum((pi_[ok] - pj[ok]) ** 2); dn = np.sum(pi_[ok] * (1 - pj[ok]) + pj[ok] * (1 - pi_[ok]))
            if dn > 0: D[i, j] = D[j, i] = -np.log(1.0 - min(num / dn, 0.999999))
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
            i, j = int(row[0]), int(row[1]); l = frozenset([i]) if i < nl else cl[i]; r = frozenset([j]) if j < nl else cl[j]; cl[nl + k] = frozenset(l | r)
        return cl
    co = clusters(Zo); sc = np.zeros(len(Zo)); rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.choice(n_snp, n_snp, replace=True); Zb = linkage(squareform(_reynolds_D(gt[:, idx], labels, pops), checks=False), method="average"); cb = set(clusters(Zb).values())
        for k, cl in co.items():
            if cl in cb: sc[k - nl] += 1
    supp = (sc / n_boot * 100).round(0); tr = to_tree(Zo)
    def nw(node):
        if node.is_leaf(): return f"{pops[node.id]}"
        l = nw(node.get_left()); r = nw(node.get_right())
        return f"({l},{r}){int(supp[node.id - nl])}:{node.dist:.4f}" if node.id >= nl else f"({l},{r}):{node.dist:.4f}"
    return {"tree": tr, "Z": Zo, "newick": nw(tr) + ";", "support": supp, "D": Do, "pops": pops}

@cache_data
def midpoint_info(D, labels):
    D = np.asarray(D, dtype=float); n = len(labels); D = (D + D.T) / 2; np.fill_diagonal(D, 0); im, jm, md = 0, 1, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] > md: md = D[i, j]; im, jm = i, j
    return {"leaf_i": labels[im], "leaf_j": labels[jm], "max_dist": float(md), "midpoint": float(md / 2)}

# ============================================================
# GWAS
# ============================================================
def _logistic_irls(X, y, max_iter=25, tol=1e-6):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30); mu = 1.0 / (1.0 + np.exp(-eta)); W = mu * (1 - mu) + 1e-9; H = (X.T * W) @ X + 1e-8 * np.eye(X.shape[1])
        try: step = np.linalg.solve(H, X.T @ (y - mu))
        except np.linalg.LinAlgError: return float(beta[-1]), None
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol: beta = beta_new; break
        beta = beta_new
    eta = np.clip(X @ beta, -30, 30); mu = 1.0 / (1.0 + np.exp(-eta)); W = mu * (1 - mu) + 1e-9
    try: cov = np.linalg.inv((X.T * W) @ X + 1e-8 * np.eye(X.shape[1])); se = float(np.sqrt(cov[-1, -1]))
    except np.linalg.LinAlgError: return float(beta[-1]), None
    return float(beta[-1]), se

def _pairwise_core(gt, snp_df, groups, ga, gb, maf_min=0.05, method="allelic", covs=None):
    groups = np.asarray(groups).astype(str); mask = np.isin(groups, [ga, gb])
    if mask.sum() < 20: raise ValueError(f"<20 individus pour {ga} vs {gb}")
    g = gt[mask]; y = (groups[mask] == ga).astype(int)
    if (y == 1).sum() < 5 or (y == 0).sum() < 5: raise ValueError("Moins de 5 individus dans une des classes")
    snp = snp_df.reset_index(drop=True); n = min(g.shape[1], len(snp)); g = g[:, :n]; snp = snp.iloc[:n].reset_index(drop=True); cv = covs[mask] if covs is not None else None
    gA, gB = g[y == 1], g[y == 0]; nA = (~np.isnan(gA)).sum(0).astype(float); nB = (~np.isnan(gB)).sum(0).astype(float); sA = np.nansum(gA, 0); sB = np.nansum(gB, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pA = sA / (2 * nA); pB = sB / (2 * nB); tot = (sA + sB) / (2 * (nA + nB)); mj = np.minimum(tot, 1 - tot)
        hs = (2 * pA * (1 - pA) * nA + 2 * pB * (1 - pB) * nB) / (nA + nB); pb = (pA * nA + pB * nB) / (nA + nB); ht = 2 * pb * (1 - pb)
        fv = np.where(ht > 1e-9, np.clip((ht - hs) / ht, 0, None), 0.0)
    valid = (nA >= 5) & (nB >= 5) & np.isfinite(mj) & (mj >= maf_min)
    if method == "allelic":
        a1, a0, b1, b0 = sA, 2 * nA - sA, sB, 2 * nB - sB; N = a1 + a0 + b1 + b0
        with np.errstate(invalid="ignore", divide="ignore"):
            den = (a1 + a0) * (b1 + b0) * (a1 + b1) * (a0 + b0); chi2 = N * (a1 * b0 - a0 * b1) ** 2 / den
        valid &= np.isfinite(chi2) & (den > 0); pv = np.where(valid, stats.chi2.sf(np.where(valid, chi2, 0), 1), np.nan)
    else:
        chi2 = np.full(n, np.nan); pv = np.full(n, np.nan)
        for j in np.where(valid)[0]:
            X = np.column_stack([np.ones(len(y))] + ([cv] if cv is not None else []) + [g[:, j]]); val = ~np.isnan(X).any(1)
            b_, se = _logistic_irls(X[val], y[val])
            if se is None or se <= 0: valid[j] = False; continue
            z = b_ / se; chi2[j] = z ** 2; pv[j] = float(2 * stats.norm.sf(abs(z)))
        valid &= np.isfinite(pv)
    idx = np.where(valid)[0]
    if len(idx) == 0: return pd.DataFrame()
    df = pd.DataFrame({"SNP": snp["SNP"].values[idx], "CHR": snp["CHR"].astype(str).values[idx], "BP": snp["BP"].astype(int).values[idx], "freq_A": pA[idx].round(4), "freq_B": pB[idx].round(4), "delta_freq": (pA - pB)[idx].round(4), "MAF": mj[idx].round(4), "chi2": chi2[idx].round(4), "P": pv[idx], "FST_snp": fv[idx].round(4), "n_A": nA[idx].astype(int), "n_B": nB[idx].astype(int)})
    return df.sort_values("P").reset_index(drop=True)

@cache_data
def gwas_pairwise(gt, snp_df, labels, ra, rb, maf_min=0.05, method="allelic", covs=None): return _pairwise_core(gt, snp_df, labels, ra, rb, maf_min, method, covs)

@cache_data
def gwas_pairwise_group(gt, snp_df, groups, ga, gb, maf_min=0.05, method="allelic", covs=None): return _pairwise_core(gt, snp_df, groups, ga, gb, maf_min, method, covs)

@cache_data
def get_pairs(labels): r = sorted(pd.unique(labels).astype(str)); return r, [(r[i], r[j]) for i in range(len(r)) for j in range(i + 1, len(r))]

def _vc_delta(Y, X0, K):
    n = len(Y); d, U = np.linalg.eigh(K); d = np.maximum(d, 0); Yt = U.T @ Y; X0t = U.T @ X0
    def nll(dl):
        if dl <= 0: return 1e18
        w = 1.0 / np.sqrt(dl * d + 1.0); Yw = Yt * w; Xw = X0t * w[:, None]
        try: b, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
        except Exception: return 1e18
        s2 = float(np.sum((Yw - Xw @ b) ** 2) / n)
        return 1e18 if s2 <= 0 else 0.5 * (n * np.log(s2) + np.sum(np.log(dl * d + 1.0)))
    grid = np.logspace(-3, 3, 30); lls = np.array([nll(dl) for dl in grid]); ib = int(np.argmin(lls))
    try: return max(float(minimize_scalar(nll, bounds=(grid[max(0, ib - 1)], grid[min(29, ib + 1)]), method="bounded").x), 1e-6)
    except Exception: return max(float(grid[ib]), 1e-6)

_GWAS_COLS = ["SNP_idx", "MAF", "BETA", "SE", "T", "P"]

@cache_data
def run_gwas(gt, pheno, K, covs, maf_min=0.05, miss_max=0.05, use_lmm=True, is_bin=False):
    pheno = np.asarray(pheno, dtype=float); valid = np.isfinite(pheno)
    if valid.sum() < 10: raise ValueError("<10 individus valides")
    Y = pheno[valid]; gv = gt[valid]; n = len(Y); miss = np.isnan(gv).mean(0); Gi = impute_mean(gv); p = Gi.mean(0) / 2.0; mj = np.minimum(p, 1 - p)
    keep = np.where((miss <= miss_max) & (mj >= maf_min) & (Gi.std(0) > 1e-8))[0]
    if len(keep) == 0: return pd.DataFrame(columns=_GWAS_COLS)
    X0 = np.ones((n, 1))
    if covs is not None and np.size(covs) > 0:
        C = np.asarray(covs, dtype=float)[valid]; C = C[:, np.nanstd(C, 0) > 1e-10]; X0 = np.column_stack([X0, C])
    if is_bin:
        rows = []
        for j in keep:
            b_, se = _logistic_irls(np.column_stack([X0, Gi[:, j].astype(float)]), Y)
            if se is None or se <= 0: continue
            tv = b_ / se; rows.append({"SNP_idx": int(j), "MAF": float(mj[j]), "BETA": b_, "SE": se, "T": float(tv), "P": float(2 * stats.norm.sf(abs(tv)))})
        return pd.DataFrame(rows, columns=_GWAS_COLS)
    lmm = use_lmm and isinstance(K, np.ndarray) and K.shape[0] == gt.shape[0]
    if lmm:
        Kv = K[np.ix_(valid, valid)]; dl = _vc_delta(Y, X0, Kv); d, U = np.linalg.eigh(Kv); d = np.maximum(d, 0); w = 1.0 / np.sqrt(dl * d + 1.0)
        Yt = (U.T @ Y) * w; X0t = (U.T @ X0) * w[:, None]
    else: Yt, X0t, U, w = Y, X0, None, None
    dof = n - X0t.shape[1] - 1
    if dof < 5: raise ValueError("Trop peu d'individus pour ce nombre de covariables")
    Q, _ = np.linalg.qr(X0t); Yr = Yt - Q @ (Q.T @ Yt); yy = float(Yr @ Yr); out = []
    for s in range(0, len(keep), 4000):
        idx = keep[s:s + 4000]; Gc = Gi[:, idx].astype(np.float64); Gt = (U.T @ Gc) * w[:, None] if lmm else Gc
        Gr = Gt - Q @ (Q.T @ Gt); gg = (Gr ** 2).sum(0); ok = gg > 1e-10; gg_s = np.where(ok, gg, 1.0)
        beta = (Gr.T @ Yr) / gg_s; s2 = np.clip((yy - beta ** 2 * gg_s) / dof, 1e-300, None); se = np.sqrt(s2 / gg_s); tv = beta / se; pv = 2 * t_dist.sf(np.abs(tv), df=dof)
        out.append(pd.DataFrame({"SNP_idx": idx[ok], "MAF": mj[idx][ok], "BETA": beta[ok], "SE": se[ok], "T": tv[ok], "P": pv[ok]}))
    return pd.concat(out, ignore_index=True)

def annotate_gwas(r, snp_df):
    if r.empty: return r
    idx = r["SNP_idx"].astype(int).values; r = r.copy()
    r["SNP"] = snp_df["SNP"].astype(str).values[idx]; r["CHR"] = snp_df["CHR"].astype(str).values[idx]; r["BP"] = snp_df["BP"].astype(float).astype(np.int64).values[idx]
    r["CM"] = snp_df["CM"].astype(float).values[idx] if "CM" in snp_df.columns else 0.0
    return r[["SNP_idx", "SNP", "CHR", "BP", "CM", "MAF", "BETA", "SE", "T", "P"]]

def lambda_gc(pvals):
    p = np.asarray(pvals); p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10: return np.nan
    return float(np.median(stats.chi2.isf(p, df=1)) / stats.chi2.ppf(0.5, df=1))

def _bh(p):
    p = np.asarray(p, dtype=float); n = len(p); o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1); return out

def postprocess_gwas(r, apply_gc=True):
    if r is None or r.empty: return r
    r = r.copy(); r["FDR"] = _bh(r["P"].values); r["log10P"] = -np.log10(r["P"].clip(1e-300, 1))
    if apply_gc:
        lam = lambda_gc(r["P"].values)
        if np.isfinite(lam) and lam > 1.0: r["P_GC"] = stats.chi2.sf(stats.chi2.isf(r["P"].clip(1e-300, 1), df=1) / lam, df=1); r["log10P_GC"] = -np.log10(r["P_GC"].clip(1e-300, 1))
    return r.sort_values("P").reset_index(drop=True)

def fine_mapping(r, window_mb=1.0, w_prior=0.04):
    if r is None or r.empty: return pd.DataFrame()
    d = r[np.isfinite(r["BETA"]) & np.isfinite(r["SE"]) & (r["SE"] > 0)].copy()
    if d.empty: return d
    lead = d.sort_values("P").iloc[0]; reg = d[(d["CHR"].astype(str) == str(lead["CHR"])) & ((d["BP"] - lead["BP"]).abs() <= window_mb * 1e6)].copy()
    V = reg["SE"] ** 2; z2 = reg["BETA"] ** 2 / V; reg["logABF"] = 0.5 * np.log(V / (V + w_prior)) + 0.5 * z2 * w_prior / (V + w_prior)
    reg["ABF"] = np.exp(reg["logABF"] - reg["logABF"].max()); reg["PIP"] = reg["ABF"] / reg["ABF"].sum()
    reg = reg.sort_values("PIP", ascending=False).reset_index(drop=True); reg["CUM_PIP"] = reg["PIP"].cumsum(); return reg

def run_gblup(gt, pheno, kf=5, ids=None, seed=42):
    pheno = np.asarray(pheno, dtype=float); valid = np.isfinite(pheno)
    if valid.sum() < 20: raise ValueError("<20 phénotypes")
    Y = pheno[valid]; n = len(Y); kf = max(2, min(kf, n - 1)); X = impute_mean(gt[valid]); p = X.mean(0) / 2.0; Z = X - 2 * p
    G = (Z @ Z.T) / (2 * np.sum(p * (1 - p))) + 1e-5 * np.eye(n); d, U = np.linalg.eigh(G); d = np.maximum(d, 0)
    mu = Y.mean(); Yt = U.T @ (Y - mu)
    def ll(h2): lam = (1 - h2) / h2; return -0.5 * (np.sum(np.log(d + lam)) + n * np.log(np.sum(Yt ** 2 / (d + lam))))
    grid = np.linspace(0.01, 0.99, 50); ib = int(np.argmax([ll(h) for h in grid]))
    try: h2 = float(minimize_scalar(lambda h: -ll(h), bounds=(grid[max(0, ib - 1)], grid[min(49, ib + 1)]), method="bounded").x)
    except Exception: h2 = float(grid[ib])
    lam = (1 - h2) / h2; gebv = U @ (d / (d + lam) * Yt)
    rng = np.random.default_rng(seed); idx = rng.permutation(n); folds = np.array_split(idx, kf); yp = np.zeros(n)
    for te in folds:
        tr = np.setdiff1d(idx, te); mtr = Y[tr].mean(); yp[te] = mtr + G[np.ix_(te, tr)] @ np.linalg.solve(G[np.ix_(tr, tr)] + lam * np.eye(len(tr)), Y[tr] - mtr)
    corr = float(np.corrcoef(Y, yp)[0, 1]) if n > 2 else 0.0
    names = np.asarray(ids)[valid] if ids is not None else np.where(valid)[0]
    return pd.DataFrame({"IID": names, "Phenotype": Y, "GEBV": gebv}), {"h2": h2, "corr": corr, "r2": corr ** 2, "rmse": float(np.sqrt(np.mean((Y - yp) ** 2)))}

def calculate_prs(gwas_res, gt, ind_df, top_n=50):
    if gwas_res is None or gwas_res.empty: return None
    top = gwas_res.nsmallest(top_n, "P")
    idx = top["SNP_idx"].astype(int).values
    betas = top["BETA"].values
    X = gt[:, idx]
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    individual_scores = np.dot(X, betas)
    df = pd.DataFrame({"FID": ind_df["FID"].values, "PRS": individual_scores})
    return df.groupby("FID")["PRS"].mean().sort_values(ascending=False).reset_index()

# ============================================================
# MÉTA-ANALYSE & API
# ============================================================
def _merge_res(res_race, res_group): return res_race[["SNP", "CHR", "BP", "P"]].rename(columns={"P": "P_race"}).merge(res_group[["SNP", "P"]].rename(columns={"P": "P_group"}), on="SNP", how="inner")

def combine_pvalues_fisher(res_race, res_group):
    if res_race is None or res_group is None: return None
    m = _merge_res(res_race, res_group)
    if m.empty: return None
    m["P_combined"] = [stats.combine_pvalues(r, method="fisher")[1] for r in m[["P_race", "P_group"]].values]; m["logP_combined"] = -np.log10(m["P_combined"].clip(1e-300, 1))
    return m.sort_values("P_combined").reset_index(drop=True)

def combine_stouffer(res_race, res_group, n_race, n_group):
    if res_race is None or res_group is None: return None
    m = _merge_res(res_race, res_group)
    if m.empty: return None
    wr, wg = np.sqrt(n_race), np.sqrt(n_group); zr = stats.norm.isf(m["P_race"].clip(1e-300, 1).values / 2); zg = stats.norm.isf(m["P_group"].clip(1e-300, 1).values / 2)
    m["P_combined"] = 2 * stats.norm.sf(np.abs((wr * zr + wg * zg) / np.sqrt(wr ** 2 + wg ** 2))); m["logP_combined"] = -np.log10(m["P_combined"].clip(1e-300, 1))
    return m.sort_values("P_combined").reset_index(drop=True)

@cache_data
def permutation_test_race(gt, snp_df, labels, ra, rb, n_perm=100, seed=42, max_snp=5000, maf_min=0.05):
    rng = np.random.default_rng(seed); labels = np.asarray(labels).astype(str)
    if len(snp_df) > max_snp: idx = np.sort(rng.choice(len(snp_df), max_snp, replace=False)); snp_df = snp_df.iloc[idx].reset_index(drop=True); gt = gt[:, idx]
    original = _pairwise_core(gt, snp_df, labels, ra, rb, maf_min)
    if original.empty: return pd.DataFrame(), 1.0, 0, [], 1.0
    n_obs = int((original["P"] < 0.05 / len(original)).sum()); min_obs = original["P"].min()
    mask = np.isin(labels, [ra, rb]); lab_sub = labels[mask].copy(); gt_sub = gt[mask]; counts, minps = [], []
    for _ in range(n_perm):
        rng.shuffle(lab_sub); r = _pairwise_core(gt_sub, snp_df, lab_sub, ra, rb, maf_min)
        counts.append(int((r["P"] < 0.05 / len(r)).sum()) if not r.empty else 0); minps.append(float(r["P"].min()) if not r.empty else 1.0)
    return original, float((np.sum(np.array(counts) >= n_obs) + 1) / (n_perm + 1)), n_obs, counts, float((np.sum(np.array(minps) <= min_obs) + 1) / (n_perm + 1))

_SEVERITY = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}

def traduire_consequence_vep(c):
    c = str(c).lower()
    if any(k in c for k in ("stop_gained", "frameshift", "start_lost", "stop_lost")): return "⛔ Exon (Non-sens / perte de codon)"
    if "missense" in c: return "🔴 Exon (Faux-sens - Protéine modifiée)"
    if "synonymous" in c: return "🟡 Exon (Silencieuse - Protéine inchangée)"
    if "splice" in c: return "⚠️ Jonction Exon/Intron"
    if "intron" in c: return "🟢 Intron (Régulation)"
    if "3_prime_utr" in c: return "🔵 Région 3'UTR"
    if "5_prime_utr" in c: return "🔵 Région 5'UTR"
    if "upstream" in c: return "🟣 Régulateur (Promoteur amont)"
    if "downstream" in c: return "🟣 Régulateur (Aval)"
    if "regulatory" in c or "tf_binding" in c: return "🟣 Région régulatrice"
    return "⚪ Intergénique"

def run_vep_api(items, species, batch=200):
    if not items: return pd.DataFrame()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}; results = []
    def _post(url, payload, mapping):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
            if r.status_code == 429: time.sleep(float(r.headers.get("Retry-After", 2))); r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
            if r.status_code != 200: st.warning(f"⚠️ VEP HTTP {r.status_code} : {r.text[:150]}"); return
            for item in r.json():
                sid = mapping.get(item.get("input"), item.get("input", "N/A"))
                if item.get("transcript_consequences"):
                    cons = sorted(item["transcript_consequences"], key=lambda x: _SEVERITY.get(x.get("impact", "MODIFIER"), 0), reverse=True)[0]
                    results.append({"SNP": sid, "GENE": cons.get("gene_symbol", cons.get("gene_id", "N/A")), "CONSEQUENCE": ", ".join(cons.get("consequence_terms", ["N/A"])), "IMPACT": cons.get("impact", "N/A"), "AMINO_ACID_CHANGE": cons.get("amino_acids", "N/A"), "SIFT": cons.get("sift_prediction", "N/A"), "POLYPHEN": cons.get("polyphen_prediction", "N/A")})
                elif item.get("regulatory_feature_consequences"):
                    cons = item["regulatory_feature_consequences"][0]
                    results.append({"SNP": sid, "GENE": "Région régulatrice", "CONSEQUENCE": cons.get("consequence_terms", ["N/A"])[0], "IMPACT": "MODIFIER", "AMINO_ACID_CHANGE": "N/A", "SIFT": "N/A", "POLYPHEN": "N/A"})
                else: results.append({"SNP": sid, "GENE": "Intergenic/Non-codant", "CONSEQUENCE": item.get("most_severe_consequence", "N/A"), "IMPACT": "MODIFIER", "AMINO_ACID_CHANGE": "N/A", "SIFT": "N/A", "POLYPHEN": "N/A"})
        except Exception as e: st.warning(f"⚠️ VEP réseau : {e}")
    reg = [(s, v) for s, k, v in items if k == "region"]; ids = [(s, v) for s, k, v in items if k == "id"]
    for i in range(0, len(reg), batch):
        ch = reg[i:i + batch]; _post(f"https://rest.ensembl.org/vep/{species}/region?regulatory=1", {"variants": [v for _, v in ch]}, {v: s for s, v in ch}); time.sleep(0.4)
    for i in range(0, len(ids), batch):
        ch = ids[i:i + batch]; _post(f"https://rest.ensembl.org/vep/{species}/id?regulatory=1", {"ids": [v for _, v in ch]}, {v: s for s, v in ch}); time.sleep(0.4)
    return pd.DataFrame(results)

def fetch_ncbi_gene_info(gene_list, taxon):
    results = []
    for gene in gene_list:
        url = f"https://api.ncbi.nlm.nih.gov/datasets/v2/gene/symbol/{quote(str(gene))}/taxon/{taxon}"
        try:
            r = requests.get(url, timeout=30, headers={"Accept": "application/json"}); reports = (r.json().get("reports") or []) if r.status_code == 200 else []
            gi = (reports[0].get("gene") or {}) if reports else {}
            if not gi: results.append({"GENE": gene, "Nom complet": "Non trouvé", "Chromosome": "N/A", "Résumé": "N/A"}); continue
            summ = gi.get("summary")
            if isinstance(summ, list): summ = " ".join(x.get("description", "") for x in summ if isinstance(x, dict))
            results.append({"GENE": gene, "Nom complet": gi.get("description", "N/A"), "Chromosome": (gi.get("chromosomes") or ["N/A"])[0], "Résumé": (str(summ)[:200] + "...") if summ else "N/A"})
        except Exception as e: results.append({"GENE": gene, "Nom complet": "Erreur API", "Chromosome": "N/A", "Résumé": str(e)})
        time.sleep(0.4)
    return pd.DataFrame(results)

def fetch_uniprot_id(gene_name, taxon):
    q = f"gene:{gene_name} AND organism_id:{taxon}"; url = f"https://rest.uniprot.org/uniprotkb/search?query={quote(q)}&format=json&size=1"
    try:
        r = requests.get(url, timeout=30)
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
    sty = {style: ({"color": "spectrum"} if color_scheme == "spectrum" else {"colorscheme": color_scheme})}; hl = ""
    if highlight_res and str(highlight_res).strip().isdigit(): hl = f"viewer.addStyle({{resi: {int(highlight_res)}}}, {{stick: {{colorscheme: 'redCarbon'}}}});"
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
        return s + "<br>", ref_aln
    ha, ref_aln = fmt(seq_a, name_a, "green"); hb, _ = fmt(seq_b, name_b, "blue")
    return ("<div style='font-family:monospace;font-size:16px;line-height:1.5;'>" + ha + f"<b>{'Référence':10}</b> | " + "".join(f"<span style='color:grey;'>{r}</span>" for r in ref_aln) + "<br>" + hb + "</div>")

def ortholog_lookup(chrom, bp, species, target="homo_sapiens"):
    base = "https://rest.ensembl.org"; h = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        genes = []
        for w in (0, 50000):
            r = requests.get(f"{base}/overlap/region/{species}/{chrom}:{max(1, bp - w)}-{bp + w}?feature=gene", headers=h, timeout=30); genes = r.json() if r.status_code == 200 else []
            if genes: break
        if not genes: return None
        g = next((x for x in genes if x.get("biotype") == "protein_coding"), genes[0]); gid = g["id"]
        r = requests.get(f"{base}/homology/id/{species}/{gid}?target_species={target};type=orthologues", headers=h, timeout=60); data = r.json().get("data", []) if r.status_code == 200 else []
        homs = data[0].get("homologies", []) if data else []
        if not homs: return {"gene_src": g.get("external_name", gid), "gene_id": gid, "human_gene": None}
        t = homs[0]["target"]; r = requests.get(f"{base}/lookup/id/{t['id']}", headers=h, timeout=30); info = r.json() if r.status_code == 200 else {}
        return {"gene_src": g.get("external_name", gid), "gene_id": gid, "human_gene": info.get("display_name", t["id"]), "human_id": t["id"], "chr": info.get("seq_region_name"), "start": info.get("start"), "end": info.get("end"), "perc_id": t.get("perc_id"), "assembly": info.get("assembly_name")}
    except Exception: return None

def annotate_genes(snp_df, gff, wk=50):
    if gff is None or gff.empty: return pd.DataFrame()
    w = wk * 1000; res = []; gchr = gff["CHR"].astype(str).values
    for _, s in snp_df.iterrows():
        pos = int(s.get("BP", s.get("START", 0))); ch = str(s["CHR"]); ng = gff[(gchr == ch) & (gff["START"] <= pos + w) & (gff["END"] >= pos - w)]
        for _, g in ng.iterrows():
            d = 0 if g["START"] <= pos <= g["END"] else min(abs(g["START"] - pos), abs(g["END"] - pos))
            res.append({"SNP": s.get("SNP", f"{ch}:{pos}"), "CHR": ch, "SNP_POS": pos, "GENE": g["GENE"], "GENE_START": g["START"], "GENE_END": g["END"], "DIST_KB": round(d / 1000.0, 2)})
    return pd.DataFrame(res)

def annotate_genes_cm(snp_df, map_df, gff, window_cm=10, window_kb_fallback=100):
    if gff is None or gff.empty or snp_df is None or len(snp_df) == 0 or map_df is None: return pd.DataFrame()
    gff_chr = gff["CHR"].astype(str).values; map_chr = map_df["CHR"].astype(str).values; res = []
    for _, s in snp_df.iterrows():
        ch = str(s["CHR"]); pos = int(s.get("BP", s.get("START", 0)))
        ng = gff[gff_chr == ch]
        if ng.empty: continue
        m = map_df[map_chr == ch].sort_values("BP"); bp_arr = m["BP"].astype(float).values; cm_arr = m["CM"].astype(float).values
        gs = ng["START"].astype(float).values; ge = ng["END"].astype(float).values
        has_cm = len(m) >= 2 and float(np.ptp(cm_arr)) > 0
        if has_cm:
            snp_cm = float(np.interp(pos, bp_arr, cm_arr)); gs_cm = np.interp(gs, bp_arr, cm_arr); ge_cm = np.interp(ge, bp_arr, cm_arr)
            d_cm = np.where((gs_cm <= snp_cm) & (snp_cm <= ge_cm), 0.0, np.minimum(np.abs(gs_cm - snp_cm), np.abs(ge_cm - snp_cm)))
            keep = d_cm <= window_cm; window = f"±{window_cm} cM"
        else:
            d_cm = np.full(len(ng), np.nan); keep = (gs <= pos + window_kb_fallback * 1000) & (ge >= pos - window_kb_fallback * 1000); window = f"±{window_kb_fallback} kb (repli : pas de carte cM)"
        d_bp = np.where((gs <= pos) & (pos <= ge), 0.0, np.minimum(np.abs(gs - pos), np.abs(ge - pos)))
        for k in np.where(keep)[0]:
            res.append({"SNP": s.get("SNP", f"{ch}:{pos}"), "CHR": ch, "SNP_POS_BP": pos, "GENE": ng["GENE"].iloc[k], "GENE_START": int(gs[k]), "GENE_END": int(ge[k]),
                        "DIST_KB": round(float(d_bp[k]) / 1000.0, 2), "DIST_CM": None if np.isnan(d_cm[k]) else round(float(d_cm[k]), 3), "WINDOW": window})
    return pd.DataFrame(res)

def run_enrichment(gene_list, gene_sets=None, organism="Human", outdir=None):
    try:
        import gseapy as gp
        genes = list({str(g).strip() for g in gene_list if str(g).strip() and str(g) != "nan"})
        if len(genes) < 3: return pd.DataFrame()
        if gene_sets is None: gene_sets = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
        enr = gp.enrichr(gene_list=genes, gene_sets=gene_sets, organism=organism, outdir=outdir, cutoff=1.0)
        if enr is None or enr.results is None: return pd.DataFrame()
        df = enr.results.copy()
        for old, new in {"Term": "TERM", "Adjusted P-value": "FDR", "P-value": "PVAL", "Overlap": "OVERLAP", "Genes": "GENES", "Combined Score": "SCORE"}.items():
            if old in df.columns: df = df.rename(columns={old: new})
        return df.sort_values("FDR").reset_index(drop=True) if "FDR" in df.columns else df
    except Exception as e: st.error(f"❌ Enrichr : {e}"); return pd.DataFrame()

def query_string_network(genes, species, required_score=400):
    if not genes or len(genes) < 2: return pd.DataFrame(), pd.DataFrame()
    ids = "%0d".join([quote(str(g).strip()) for g in genes if str(g).strip()])
    url = f"https://string-db.org/api/json/network?identifiers={ids}&species={species}&required_score={required_score}&caller_identity=livestock_genomics_pipeline"
    try:
        r = requests.get(url, timeout=60); r.raise_for_status(); data = r.json()
        if not data: return pd.DataFrame(), pd.DataFrame()
        edges = pd.DataFrame(data)[["preferredName_A", "preferredName_B", "score"]].drop_duplicates(); edges.columns = ["SOURCE", "TARGET", "SCORE"]
        return edges, pd.DataFrame({"ID": sorted(set(edges["SOURCE"]).union(set(edges["TARGET"])))})
    except Exception: return pd.DataFrame(), pd.DataFrame()

# ============================================================
# EXPORTS & GRAPHIQUES
# ============================================================
def _alleles_for_export(snp_df, m):
    a_minor = snp_df["A2"].astype(str).values[:m] if "A2" in snp_df.columns else np.array(["G"] * m)
    a_major = snp_df["A1"].astype(str).values[:m] if "A1" in snp_df.columns else np.array(["A"] * m)
    return a_minor, a_major

def build_plink(gt, ind_df, snp_df, prefix="genome"):
    n, m = gt.shape; nb = (n + 3) // 4; code = np.full(gt.shape, 0b01, dtype=np.uint8)
    code[gt == 0] = 0b11; code[gt == 1] = 0b10; code[gt == 2] = 0b00
    code = np.vstack([code, np.zeros((nb * 4 - n, m), dtype=np.uint8)]).reshape(nb, 4, m)
    packed = code[:, 0, :] | (code[:, 1, :] << 2) | (code[:, 2, :] << 4) | (code[:, 3, :] << 6)
    a1, a2 = _alleles_for_export(snp_df, m); clean = lambda a: np.where(np.isin(a, ["", ".", "nan", "None"]), "0", a)
    bim = pd.DataFrame({"c": snp_df["CHR"].values[:m], "s": snp_df["SNP"].values[:m], "cm": snp_df["CM"].values[:m], "bp": snp_df["BP"].astype(int).values[:m], "a1": clean(a1), "a2": clean(a2)})
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}.bed", bytes([0x6C, 0x1B, 0x01]) + np.ascontiguousarray(packed.T).tobytes())
        zf.writestr(f"{prefix}.bim", bim.to_csv(sep="\t", header=False, index=False))
        zf.writestr(f"{prefix}.fam", "\n".join(f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9" for _, r in ind_df.iterrows()) + "\n")
    return bio.getvalue()

def build_vcf(gt, ind_df, snp_df, proj="Genome"):
    m = gt.shape[1]; alt, ref = _alleles_for_export(snp_df, m); acgt = np.array(["A", "C", "G", "T"])
    ref = np.where(np.isin(ref, acgt), ref, "A"); alt = np.where(np.isin(alt, acgt), alt, "G")
    h = ["##fileformat=VCFv4.2", f"##source={proj}", '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">', "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(ind_df["IID"].astype(str))]
    lut = np.array(["0/0", "0/1", "1/1"]); strs = np.where(np.isnan(gt), "./.", lut[np.nan_to_num(gt, nan=0).astype(int).clip(0, 2)]); b = []
    for j in range(m):
        r = snp_df.iloc[j]; b.append(f"{r['CHR']}\t{int(r['BP'])}\t{r['SNP']}\t{ref[j]}\t{alt[j]}\t.\tPASS\t.\tGT\t" + "\t".join(strs[:, j]))
    return "\n".join(h + b) + "\n"

def plot_missing(mi, ms):
    f = make_subplots(rows=1, cols=2, subplot_titles=("Missing/Ind", "Missing/SNP")); f.add_trace(go.Histogram(x=mi, nbinsx=60, marker_color="skyblue"), 1, 1); f.add_trace(go.Histogram(x=ms, nbinsx=60, marker_color="coral"), 1, 2)
    f.update_layout(height=400, showlegend=False); return f

def plot_pca(s, v, l):
    f = px.scatter(pd.DataFrame({"PC1": s[:, 0], "PC2": s[:, 1], "Pop": l}), x="PC1", y="PC2", color="Pop", title=f"PCA PC1({v[0]:.1f}%) vs PC2({v[1]:.1f}%)", height=550)
    f.update_traces(marker=dict(size=10, line=dict(width=1, color="white"))); return f

def plot_mds(m, l):
    f = px.scatter(pd.DataFrame({"MDS1": m[:, 0], "MDS2": m[:, 1], "Pop": list(l)}), x="MDS1", y="MDS2", color="Pop", title="MDS (distance IBS)", height=520)
    f.update_traces(marker=dict(size=11, line=dict(width=1, color="white"))); return f

def _cum_positions(df, poscol):
    df = df.assign(_k=df["CHR"].map(_chk_num), _s=df["CHR"].astype(str)).sort_values(["_k", "_s", poscol], kind="stable").reset_index(drop=True); offs = {}; ticks = []; cur = 0.0
    for c, sub in df.groupby("CHR", sort=False):
        mx = float(sub[poscol].max()) / 1e6; offs[c] = cur; ticks.append((str(c), cur + mx / 2)); cur += mx + 5
    df["_x"] = df["CHR"].map(offs) + df[poscol].astype(float) / 1e6; return df, ticks

def plot_manhattan(df, yc="P", title="", thr=None):
    df = df.copy(); ylab = yc
    if yc in ("P", "P_GC"): df["_y"] = -np.log10(np.clip(df[yc], 1e-300, 1)); yc = "_y"; ylab = "-log10(P)"
    df = df[np.isfinite(df[yc])]
    if df.empty: return go.Figure()
    df, ticks = _cum_positions(df, "BP"); fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR", sort=False)): fig.add_trace(go.Scatter(x=sub["_x"], y=sub[yc], mode="markers", marker=dict(size=5, color="#2c3e50" if i % 2 == 0 else "#7f8c8d"), text=sub["SNP"], hoverinfo="text+y", showlegend=False))
    if thr: fig.add_hline(y=thr, line_dash="dash", line_color="red")
    fig.update_layout(title=title, xaxis=dict(title="Chromosome", tickmode="array", tickvals=[t[1] for t in ticks], ticktext=[t[0] for t in ticks]), yaxis_title=ylab, height=500); return fig

def plot_qq(pvals, title="QQ-plot"):
    p = np.sort(np.asarray(pvals)[np.isfinite(pvals)]); p = p[(p > 0) & (p <= 1)]; n = len(p)
    if n == 0: return go.Figure()
    exp = -np.log10((np.arange(1, n + 1) - 0.5) / n); obs = -np.log10(p)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=exp, y=obs, mode="markers", marker=dict(size=4), name="obs")); fig.add_trace(go.Scatter(x=[0, exp.max()], y=[0, exp.max()], mode="lines", line=dict(color="red", dash="dash"), name="attendu"))
    fig.update_layout(title=title, xaxis_title="-log10(P) attendu", yaxis_title="-log10(P) observé", height=420); return fig

def plot_ld(ld):
    if ld is None or ld.empty: return None
    return px.line(ld.sort_values("dist_kb"), x="dist_kb", y="r2", markers=True, title="Décroissance du LD (r²)")

def plot_admix(Q, labs, pops, K):
    df = pd.DataFrame(Q, columns=[f"K{k+1}" for k in range(K)]); df["IID"] = labs; df["Pop"] = pops
    df["_d"] = Q.argmax(1); df["_po"] = pd.Categorical(pops, categories=sorted(set(pops)))
    df = df.sort_values(["_po","_d"]).reset_index(drop=True); df["x"] = np.arange(len(df))
    fig = go.Figure()
    for k in range(K):
        fig.add_trace(go.Bar(x=df["x"], y=df[f"K{k+1}"], name=f"K{k+1}", marker_color=px.colors.qualitative.Set2[k%8]))
    fig.update_layout(barmode="stack", title=f"Admixture K={K}", height=500)
    return fig

def plot_gblup(df):
    fig = px.scatter(df, x="Phenotype", y="GEBV", title="GBLUP: Prédictions vs Réel")
    fig.update_traces(marker=dict(size=10, color="royalblue"))
    if len(df) > 1:
        try:
            z = np.polyfit(df["Phenotype"], df["GEBV"], 1); p = np.poly1d(z); xr = np.linspace(df["Phenotype"].min(), df["Phenotype"].max(), 100)
            fig.add_trace(go.Scatter(x=xr, y=p(xr), mode="lines", name="OLS", line=dict(color="red", dash="dash")))
        except Exception: pass
    return fig

def plot_pip(df):
    return px.bar(df.head(20), x="SNP", y="PIP", title="PIP (Posterior Inclusion Probability)", color="PIP", color_continuous_scale="Viridis")

def plot_win_manh(df, vc='MEAN_FST', title="", sigma=3.0):
    df = df[np.isfinite(df[vc])].copy()
    if df.empty: return None
    df = df.assign(_k=df['CHR'].map(_chk)).sort_values(['_k','MID']).reset_index(drop=True)
    fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby('CHR')):
        fig.add_trace(go.Scatter(x=sub['MID']/1e6, y=sub[vc], mode='markers',
            marker=dict(size=7, color='#2c3e50' if i%2==0 else '#7f8c8d', line=dict(width=0.5, color='white')),
            text=[f"{c}:{int(m/1e6)}Mb<br>{vc}={v:.4f}" for c, m, v in zip(sub['CHR'], sub['MID'], sub[vc])],
            hoverinfo='text', showlegend=False))
    mu, sd = df[vc].mean(), df[vc].std()
    fig.add_hline(y=mu+sigma*sd, line_dash='dash', line_color='red', annotation_text=f'μ+{sigma}σ')
    fig.update_layout(title=title, xaxis_title='Position (Mb)', yaxis_title=vc, height=520)
    return fig

def plot_gene_track(si, gff, wk=50):
    if gff is None or gff.empty: return None
    ch = si["CHR"]; pos = si.get("BP", si.get("START", 0)); w = wk*1000
    rg = gff[(gff["CHR"]==ch) & (gff["START"]<=pos+w) & (gff["END"]>=pos-w)]
    fig = go.Figure()
    for i, (_, g) in enumerate(rg.iterrows()):
        fig.add_trace(go.Scatter(x=[g["START"]/1e6, g["END"]/1e6], y=[i,i], mode="lines",
                                 line=dict(width=10, color="royalblue"), name=g["GENE"],
                                 text=[g["GENE"], g["GENE"]], hoverinfo="text"))
    fig.add_vline(x=pos/1e6, line_dash="dash", line_color="red", annotation_text=f"SNP: {si.get('SNP','')}")
    fig.update_layout(title=f"Gènes CHR{ch}", xaxis_title="Position (Mb)", yaxis=dict(showticklabels=False), height=400+len(rg)*20)
    return fig

def plot_roh_islands(df):
    if df is None or df.empty: return None
    d = df.copy(); d["_k"] = d["CHR"].map(_chk)
    d = d.sort_values(["_k", "MID"]).reset_index(drop=True)
    fig = go.Figure()
    for c, sub in d.groupby("CHR"):
        fig.add_trace(go.Scatter(
            x=sub["MID"] / 1e6, y=sub["FRAC_POP"], mode="markers",
            marker=dict(size=sub["N_ROH"] * 0.8 + 5, color=sub["FRAC_POP"], colorscale="Viridis", line=dict(width=0.5, color="white")),
            text=[f"CHR{c}:{int(m/1e6)}Mb<br>{int(n)} ROH<br>Frac={f:.1%}<br>{l:.0f} kb" for m, n, f, l in zip(sub["MID"], sub["N_ROH"], sub["FRAC_POP"], sub["LENGTH_KB"])],
            hoverinfo="text", showlegend=False))
    fig.update_layout(title="ROH Islands (fenêtres de 100 kb)", xaxis_title="Position (Mb)", yaxis_title="Fraction population en ROH", height=520)
    return fig

def render_diagnostic_dashboard(ind_df, snp_df, froh, roh_df, gwas_res, ne_dict, fst_val):
    st.markdown("## 🩺 Diagnostic Dashboard")
    st.caption("🟢 OK · 🟡 Attention · 🔴 Alerte — synthèse par population.")
    pops = sorted(ind_df["FID"].astype(str).unique())
    rows = []
    for pop in pops:
        mask = (ind_df["FID"].astype(str) == pop).values
        n_ind = int(mask.sum())
        f_mean = np.nan
        if froh is not None and len(froh) == len(ind_df): f_mean = float(np.nanmean(froh[mask]))
        if np.isfinite(f_mean):
            if f_mean < 0.05: f_color, f_disp = "🟢", f"{f_mean:.4f}"
            elif f_mean < 0.15: f_color, f_disp = "🟡", f"{f_mean:.4f}"
            else: f_color, f_disp = "🔴", f"{f_mean:.4f}"
        else: f_color, f_disp = "⚪", "N/A"
        ne_val = None
        if ne_dict is not None and pop in ne_dict:
            ndf = ne_dict[pop]
            if ndf is not None and not ndf.empty: ne_val = float(ndf["Ne"].mean())
        ne_disp = f"{ne_val:.0f}" if ne_val is not None else "N/A"
        n_hits = 0
        if gwas_res is not None and not gwas_res.empty and "P" in gwas_res.columns:
            n_hits = int((gwas_res["P"] < 0.05 / len(gwas_res)).sum())
        rows.append({"Population": pop, "N": n_ind, "FROH": f"{f_color} {f_disp}", "Ne moyen": ne_disp, "GWAS hits": n_hits})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.markdown("### 🚦 Alertes globales")
    c1, c2, c3 = st.columns(3)
    if froh is not None and len(froh) > 0:
        avg_f = float(np.nanmean(froh))
        delta = "🔴 élevé" if avg_f > 0.15 else "🟡 modéré" if avg_f > 0.05 else "🟢 faible"
        c1.metric("FROH moyen", f"{avg_f:.4f}", delta=delta)
    if gwas_res is not None and not gwas_res.empty:
        n_sig = int((gwas_res["P"] < 0.05 / len(gwas_res)).sum())
        c2.metric("GWAS Bonferroni", n_sig, delta="🔴 signal" if n_sig > 0 else "🟢 aucun")
    if ne_dict:
        ne_vals = [float(d["Ne"].mean()) for d in ne_dict.values() if d is not None and not d.empty]
        if ne_vals:
            mean_ne = np.mean(ne_vals)
            delta = "🔴 faible" if mean_ne < 100 else "🟢 OK"
            c3.metric("Ne moyen", f"{mean_ne:.0f}", delta=delta)

# ============================================================
# STREAMLIT MAIN
# ============================================================
_KEYS = ["gt","ind_df","snp_df","gt_filt","ind_filt","snp_filt","gt_pre","ind_pre","snp_pre",
         "gt_pruned","snp_pruned","qc_stats","pca_s","pca_v","mds_c","king_K","fst","fst_M",
         "fst_pops","admix_Q","admix_K","cv_res","ld_df","roh_df","froh","ne_dict","reyn_D",
         "reyn_pops","pheno_df","gwas_res","gwas_pheno","lambda_gc","gebv_df","gblup_stats",
         "ann_df","_plink","_vcf","mapping_df","fst_snp_df","win_fst","taj_df","taj_race",
         "css_df","pair_res","pair_res_race","pair_res_group","pair_name_race","pair_name_group",
         "nj_result","nj_mid","sel_gff","sel_ann","sel_ann_race","sel_ann_group","scan_df",
         "vep_res","ncbi_gene_res","lift_res","ihs_res","xpehh_res","strat_sel_df","shannon_df","shannon_race"]

def init_state():
    for k in _KEYS:
        if k not in st.session_state: st.session_state[k] = None

def main():
    init_state()
    st.title("🧬 Universal Livestock Genomics Pipeline v23.8")
    st.caption("Analyses génomiques de niveau publication — Bovins & Camélidés")

    if DEMO:
        st.info("🎮 Mode Démo activé : Données simulées préchargées.")
        if st.session_state.gt is None:
            gt, ind, snp = gen_demo(autosomes=SPECIES_CONFIG["Bovin (Taurin/Zébu)"]["autosomes"], ars=SPECIES_CONFIG["Bovin (Taurin/Zébu)"]["ars"])
            st.session_state.update({"gt": gt, "ind_df": rec_fid(ind), "snp_df": snp})

    with st.sidebar:
        st.header("⚙️ Configuration")
        species = st.selectbox("Espèce", list(SPECIES_CONFIG.keys()), key="species_sel")
        cfg = SPECIES_CONFIG[species]
        
        st.header("📁 Données")
        mode = st.radio("Source", ["Demo", "PED/MAP", "VCF", "Axiom→PED"], key="src")
        
        if mode == "Demo":
            if st.button("🎲 Générer démo", use_container_width=True):
                gt, i, s = gen_demo(autosomes=cfg["autosomes"], ars=cfg["ars"])
                st.session_state.update({"gt": gt, "ind_df": rec_fid(i), "snp_df": s})
                st.success("✅")
        elif mode == "PED/MAP":
            ped = st.file_uploader(".ped", type=["ped", "gz"])
            mp = st.file_uploader(".map", type=["map", "gz"])
            if ped and mp and st.button("📥 Charger"):
                try:
                    mdf = parse_map(mp.read()); gt, ind, a1, a2 = parse_ped(ped.read(), len(mdf))
                    if len(mdf) > gt.shape[1]: mdf = mdf.iloc[:gt.shape[1]].reset_index(drop=True)
                    elif len(mdf) < gt.shape[1]: gt = gt[:, :len(mdf)]
                    mdf["A1"] = a1; mdf["A2"] = a2
                    st.session_state.update({"gt": gt, "ind_df": rec_fid(ind), "snp_df": mdf})
                    st.success(f"✅ {gt.shape[0]}×{gt.shape[1]}")
                except Exception as e: st.error(f"❌ {e}")
        elif mode == "VCF":
            vf = st.file_uploader(".vcf", type=["vcf", "gz"])
            if vf and st.button("📥 Charger"):
                try:
                    gt, ind, snp = parse_vcf(vf.read())
                    st.session_state.update({"gt": gt, "ind_df": rec_fid(ind), "snp_df": snp})
                    st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
        else:
            af = st.file_uploader("Axiom brut", type=["txt", "csv", "tsv", "gz"])
            if af and st.button("🔄 Convertir"):
                try:
                    s, g, ns = parse_axiom(af.read(), af.name)
                    mpl = [f"1\tSNP_{i+1:06d}\t0\t{i+1}" for i in range(ns)]
                    pdl = ["\t".join([sample_fid_iid(smp)[0], sample_fid_iid(smp)[1], "0", "0", "0", "-9"] +
                            [a for pr in row for a in pr]) for smp, row in zip(s, g)]
                    st.session_state["_plink_ped"] = "\n".join(pdl) + "\n"
                    st.session_state["_plink_map"] = "\n".join(mpl) + "\n"
                    st.success(f"✅ {len(s)}×{ns}")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state.get("_plink_ped"):
                st.download_button("⬇ .ped", st.session_state["_plink_ped"].encode(), "out.ped")
                st.download_button("⬇ .map", st.session_state["_plink_map"].encode(), "out.map")
        
        st.divider()
        st.header("🎛️ Seuils QC")
        geno = st.slider("Missing SNP", 0.0, 0.5, 0.05)
        mind = st.slider("Missing Ind", 0.0, 0.5, 0.05)
        maf_t = st.slider("MAF", 0.0, 0.5, 0.05)
        hwe_t = st.number_input("HWE", value=1e-6, format="%.0e")
        het_sd = st.slider("Het ±σ", 1.0, 5.0, 3.0, 0.1)
        rem_out = st.checkbox("❌ Supprimer outliers", value=True)
        ld_r2 = st.slider("LD r²", 0.05, 0.5, 0.2, 0.05)

    if st.session_state.gt is None:
        st.info("👉 Chargez des données (Demo recommandé pour tester).")
        return

    tabs = st.tabs([
        "🧹 QC", "✂️ LD", "🧬 Structure", "🎨 Admixture", "📈 Démographie",
        "🔍 Sélection", "🧪 GWAS", "🎯 Post-GWAS", "📈 GBLUP",
        "🧬 Annotation", "🧬 VEP & NCBI", "🧬 Structure 3D", "🧬 Alignement",
        "📤 Export", "🔬 Méta-analyse", "🧠 ML", "🧬 Haplotypes", "🌍 LiftOver", "🩺 Diagnostic"
    ])

    # ═══════ T0 — QC ═══════
    with tabs[0]:
        if st.button("▶ Lancer QC", key="qc_btn"):
            try:
                p = {"geno": geno, "mind": mind, "maf": maf_t, "hwe": hwe_t, "het_sd": het_sd}
                gf, inf, snf, qs = apply_qc(st.session_state.gt, st.session_state.ind_df, st.session_state.snp_df, p, cfg["autosomes"], rem_out=rem_out)
                st.session_state.update({"gt_filt": gf, "ind_filt": inf, "snp_filt": snf, "qc_stats": qs})
                st.success(f"✅ {gf.shape[0]} × {gf.shape[1]}")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.qc_stats:
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Individus", s["n_ind_final"], delta=-s["excluded_ind"])
            c2.metric("SNPs", s["n_snp_final"], delta=-s["excluded_snp"])
            c3.metric("Ind pré-outliers", s["n_ind_pre"])
            c4.metric("SNPs initiaux", s["n_snp_init"])
            st.plotly_chart(plot_missing(miss_ind(st.session_state.gt), miss_snp(st.session_state.gt)), use_container_width=True)

    # ═══════ T1 — LD ═══════
    with tabs[1]:
        if st.session_state.gt_filt is not None and st.button("▶ LD Pruning", key="ld_btn"):
            k = ld_pruning(st.session_state.gt_filt, st.session_state.snp_filt["CHR"].values, 50, 5, ld_r2)
            st.session_state.update({"gt_pruned": st.session_state.gt_filt[:, k], "snp_pruned": st.session_state.snp_filt[k].reset_index(drop=True)})
            st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNPs restants")
        if st.session_state.gt_pruned is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Avant", st.session_state.gt_filt.shape[1])
            c2.metric("Après", st.session_state.gt_pruned.shape[1])
            c3.metric("Réduction", f"{100*(1-st.session_state.gt_pruned.shape[1]/st.session_state.gt_filt.shape[1]):.1f}%")
            if st.button("▶ LD Decay", key="ld_decay_btn"):
                st.session_state.ld_df = ld_decay(st.session_state.gt_pruned, st.session_state.snp_pruned)
            if st.session_state.ld_df is not None:
                st.plotly_chart(plot_ld(st.session_state.ld_df), use_container_width=True)

    # ═══════ T2 — Structure ═══════
    with tabs[2]:
        if st.session_state.gt_filt is not None and st.button("▶ PCA + MDS + KING", key="struct_btn"):
            gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            s_, v_ = pca_analysis(gu, 10)
            st.session_state.update({"pca_s": s_, "pca_v": v_})
            st.session_state.mds_c = mds_analysis(gu, 5)
            st.session_state.king_K = king_mat(gu)
            st.success("✅")
        if st.session_state.pca_s is not None:
            st.plotly_chart(plot_pca(st.session_state.pca_s, st.session_state.pca_v, st.session_state.ind_filt["FID"].values), use_container_width=True)
            st.plotly_chart(plot_mds(st.session_state.mds_c, st.session_state.ind_filt["FID"].values), use_container_width=True)

    # ═══════ T3 — Admixture ═══════
    with tabs[3]:
        if st.session_state.gt_filt is not None:
            K = st.slider("K", 2, 10, 4, key="admix_K")
            if st.button("▶ Admixture NMF", key="admix_btn"):
                gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                Q, _ = admixture_nmf(gu, K=K)
                st.session_state.update({"admix_Q": Q, "admix_K": K})
                st.success("✅")
            if st.session_state.admix_Q is not None:
                st.plotly_chart(plot_admix(st.session_state.admix_Q, st.session_state.ind_filt["IID"].values, st.session_state.ind_filt["FID"].values, st.session_state.admix_K), use_container_width=True)

    # ═══════ T4 — Démographie ═══════
    with tabs[4]:
        if st.session_state.gt_filt is not None and st.button("▶ ROH + Ne", key="dem_btn"):
            try:
                gu = st.session_state.gt_filt; su = st.session_state.snp_filt
                roh, fr = detect_roh(gu, su["CHR"].values, su["BP"].values)
                st.session_state.update({"roh_df": roh, "froh": fr})
                ne = {}
                for f in st.session_state.ind_filt["FID"].unique():
                    m = (st.session_state.ind_filt["FID"] == f).values
                    if m.sum() >= 5: ne[str(f)] = ne_historical(gu[m], su)
                st.session_state.ne_dict = ne
                st.success(f"✅ {len(roh)} ROH trouvés.")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.froh is not None:
            df = pd.DataFrame({"FROH": st.session_state.froh, "Pop": st.session_state.ind_filt["FID"].values})
            st.plotly_chart(px.histogram(df, x="FROH", color="Pop", nbins=50, title="Distribution de FROH"), use_container_width=True)

    # ═══════ T5 — Sélection ═══════
    with tabs[5]:
        use_wc = st.checkbox("Weir & Cockerham (1984)", value=True)
        if st.session_state.gt_filt is not None and st.button("▶ FST global + UPGMA", key="fst_btn"):
            gu = st.session_state.gt_filt; lab = st.session_state.ind_filt["FID"].astype(str).values
            st.session_state.fst = fst_snp_wc(gu, lab) if use_wc else None
            M, pops = fst_pairwise(gu, lab, use_wc84=use_wc)
            st.session_state.update({"fst_M": M, "fst_pops": pops})
            D, rp = reynolds(gu, lab); st.session_state.update({"reyn_D": D, "reyn_pops": rp})
            st.success("✅")
        if st.session_state.fst is not None:
            sf = st.session_state.snp_filt; nf = min(len(st.session_state.fst), len(sf))
            dff = pd.DataFrame({"SNP": sf["SNP"].values[:nf], "CHR": sf["CHR"].values[:nf], "BP": sf["BP"].values[:nf], "FST": st.session_state.fst[:nf]})
            st.plotly_chart(plot_manhattan(dff, yc="FST", title="FST par SNP"), use_container_width=True)

        st.divider(); st.subheader("🧬 Signatures de sélection")
        if st.session_state.gt_filt is not None:
            gu_base = st.session_state.gt_filt; su = st.session_state.snp_filt; iu = st.session_state.ind_filt; lab = iu["FID"].astype(str).values
            
            ms = st.radio("Méthode", ["8.3.1 — FST par fenêtres", "8.3.2 — Tajima's D", "8.3.3 — CSS", "8.3.4 — Shannon (Diversité)"], horizontal=True)
            win_mode = st.radio("📐 Type de fenêtre", ["SNP count", "Physique (bp)"], horizontal=True)
            
            if ms.startswith("8.3.1"):
                c1, c2, c3 = st.columns(3)
                ws = c1.slider("Fenêtre (SNPs)", 10, 200, 50, key="wf")
                sp = c2.slider("Pas (SNPs)", 5, 100, 25, key="wp")
                zt = c3.slider("Seuil z", 2.0, 5.0, 3.0, 0.1, key="wz")
                if st.button("▶ Scanner FST", type="primary"):
                    w = windowed_fst(gu_base, su, lab, ws, sp)
                    w = detect_outliers(w, zt); st.session_state.win_fst = w
                    st.success(f"✅ {len(w)} fenêtres")
                if st.session_state.get("win_fst") is not None:
                    st.plotly_chart(plot_win_manh(st.session_state.win_fst, 'MEAN_FST', 'FST fenêtres'), use_container_width=True)
                    st.dataframe(st.session_state.win_fst.head(20), use_container_width=True)
            
            elif ms.startswith("8.3.2"):
                available_races = sorted(np.unique(lab))
                rs = st.selectbox("Race", available_races, key="tr")
                c1, c2 = st.columns(2)
                if win_mode.startswith("SNP"):
                    wt = c1.slider("Fenêtre (SNPs)", 10, 100, 25); stp = c2.slider("Pas (SNPs)", 5, 50, 10)
                else:
                    wt = c1.slider("Fenêtre (kb)", 100, 5000, 500, 100); stp = c2.slider("Pas (kb)", 50, 2500, 250, 50)
                if st.button(f"▶ Tajima's D — {rs}", type="primary"):
                    m = lab == rs
                    if m.sum() < 5: st.error("⚠️ < 5 individus")
                    else:
                        if win_mode.startswith("SNP"): t = tajimas_scan(gu_base[m], su, wt, stp)
                        else: t = tajimas_scan_phys(gu_base[m], su, wt, stp, min_snps=10)
                        st.session_state.taj_df = t; st.session_state.taj_race = rs
                        st.success(f"✅ {len(t)} fenêtres")
                if st.session_state.get("taj_df") is not None:
                    st.plotly_chart(plot_win_manh(st.session_state.taj_df, 'TAJIMA_D', 'Tajima D'), use_container_width=True)
                    st.dataframe(st.session_state.taj_df.sort_values('TAJIMA_D').head(20), use_container_width=True)
            
            elif ms.startswith("8.3.3"):
                if st.button("▶ CSS", type="primary"):
                    st.session_state.css_df = composite_score(gu_base, su, lab)
                    st.success(f"✅ {len(st.session_state.css_df)} SNPs")
                if st.session_state.get("css_df") is not None:
                    st.dataframe(st.session_state.css_df.head(20), use_container_width=True)
            
            elif ms.startswith("8.3.4"):
                available_races = sorted(np.unique(lab))
                rs_sh = st.selectbox("Race", available_races, key="shannon_race")
                c1, c2 = st.columns(2)
                if win_mode.startswith("SNP"):
                    wt_sh = c1.slider("Fenêtre (SNPs)", 10, 100, 25); stp_sh = c2.slider("Pas (SNPs)", 5, 50, 10)
                else:
                    wt_sh = c1.slider("Fenêtre (kb)", 100, 5000, 500, 100); stp_sh = c2.slider("Pas (kb)", 50, 2500, 250, 50)
                if st.button(f"▶ Shannon — {rs_sh}", type="primary"):
                    m = lab == rs_sh
                    if m.sum() < 5: st.error("⚠️ < 5 individus")
                    else:
                        if win_mode.startswith("SNP"): sh = shannon_scan(gu_base[m], su, wt_sh, stp_sh)
                        else: sh = shannon_scan_phys(gu_base[m], su, wt_sh, stp_sh, min_snps=10)
                        st.session_state.shannon_df = sh; st.session_state.shannon_race = rs_sh
                        st.success(f"✅ {len(sh)} fenêtres")
                if st.session_state.get("shannon_df") is not None:
                    st.plotly_chart(plot_win_manh(st.session_state.shannon_df, 'SHANNON_H', f"Shannon — {st.session_state.get('shannon_race', '')}"), use_container_width=True)
                    st.dataframe(st.session_state.shannon_df.sort_values('SHANNON_H').head(20), use_container_width=True)

    # ═══════ T6 — GWAS ═══════
    with tabs[6]:
        if st.session_state.gt_filt is not None:
            st.markdown("### 1️⃣ Phénotypes")
            pf = st.file_uploader("CSV Phénotypes", type=["csv", "txt", "gz"], key="gwas_pheno")
            if pf and st.button("📥 Charger"):
                st.session_state.pheno_df, _ = match_pheno(st.session_state.ind_filt, load_pheno(pf.read(), pf.name))
                st.success("✅")
            if st.session_state.pheno_df is not None:
                cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID", "IID", "Animal")]
                pc = st.selectbox("Phénotype", cols)
                ib = st.checkbox("Binaire (0/1)")
                if st.button("🚀 Lancer GWAS", type="primary"):
                    gg = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    sg = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    cv = st.session_state.pca_s[:, :5] if st.session_state.pca_s is not None else None
                    ps = pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce")
                    r = run_gwas(gg, ps.values, st.session_state.king_K, cv, is_bin=ib)
                    r = annotate_gwas(r, sg); r = postprocess_gwas(r)
                    st.session_state.gwas_res = r
                    st.success(f"✅ {len(r)} SNPs · λGC={lambda_gc(r['P'].values):.3f}")
                    st.plotly_chart(plot_manhattan(r, yc="P", title="Manhattan"), use_container_width=True)
                    st.dataframe(r.head(20), use_container_width=True)

    # ═══════ T7 — Post-GWAS ═══════
    with tabs[7]:
        if st.session_state.gwas_res is not None:
            st.subheader("🎯 Fine-mapping")
            pip = fine_mapping(st.session_state.gwas_res)
            st.dataframe(pip.head(10), use_container_width=True)
            st.plotly_chart(plot_pip(pip), use_container_width=True)
            
            st.divider(); st.subheader("🧬 Score Polygénique (PRS)")
            if st.button("📊 Calculer le PRS par Race"):
                gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                prs_df = calculate_prs(st.session_state.gwas_res, gu, st.session_state.ind_filt, top_n=50)
                if prs_df is not None:
                    st.plotly_chart(px.bar(prs_df, x="FID", y="PRS", title="PRS moyen par race"), use_container_width=True)
                    st.dataframe(prs_df, use_container_width=True)

    # ═══════ T8 — GBLUP ═══════
    with tabs[8]:
        if st.session_state.pheno_df is not None:
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID", "IID", "Animal")]
            pc = st.selectbox("Phénotype GBLUP", cols, key="gb_pc")
            if st.button("▶ GBLUP"):
                gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                df, stats_ = run_gblup(gu, pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce").values)
                st.plotly_chart(plot_gblup(df), use_container_width=True)
                st.json(stats_)

    # ═══════ T9 — Annotation ═══════
    with tabs[9]:
        gff_f = st.file_uploader("GFF3/GTF", type=["gff", "gff3", "gtf", "gz"], key="ann_ded_gff")
        if gff_f:
            gff = parse_gff3(gff_f.read())
            snps = st.session_state.gwas_res.sort_values("P").head(20) if st.session_state.gwas_res is not None else None
            if snps is not None and st.button("🔍 Annoter"):
                ann = annotate_genes(snps, gff, wk=50)
                st.dataframe(ann, use_container_width=True)

    # ═══════ T10 — VEP & NCBI ═══════
    with tabs[10]:
        if st.session_state.gwas_res is not None:
            snps = st.session_state.gwas_res["SNP"].head(20).tolist()
            if st.button("🔬 Lancer VEP"):
                vep_res = run_vep_api([(s, "id", s) for s in snps], cfg["vep_species"])
                st.session_state.vep_res = vep_res
                st.dataframe(vep_res, use_container_width=True)

    # ═══════ T11 — 3D ═══════
    with tabs[11]:
        st.info("Onglet 3D — Uploadez un gène VEP pour voir la structure AlphaFold.")

    # ═══════ T12 — Alignement ═══════
    with tabs[12]:
        st.info("Onglet Alignement — Comparez deux allèles pour un SNP donné.")

    # ═══════ T13 — Export ═══════
    with tabs[13]:
        if st.session_state.gt_filt is not None:
            if st.button("📦 PLINK"):
                data = build_plink(st.session_state.gt_filt, st.session_state.ind_filt, st.session_state.snp_filt, cfg["prefix"])
                st.download_button("⬇ Télécharger", data, f"{cfg['prefix']}.zip", "application/zip")
            if st.button("📄 VCF"):
                data = build_vcf(st.session_state.gt_filt, st.session_state.ind_filt, st.session_state.snp_filt, cfg["prefix"])
                st.download_button("⬇ Télécharger", data.encode(), f"{cfg['prefix']}.vcf", "text/plain")

    # ═══════ T14 — Méta-analyse ═══════
    with tabs[14]:
        st.info("Onglet Méta-analyse — Combinez des p-values (Fisher/Stouffer).")

    # ═══════ T15 — ML ═══════
    with tabs[15]:
        st.info("Onglet ML — Random Forest pour l'importance des SNPs.")

    # ═══════ T16 — Haplotypes ═══════
    with tabs[16]:
        st.info("Onglet Haplotypes — Visualisez les blocs d'haplotypes.")

    # ═══════ T17 — LiftOver ═══════
    with tabs[17]:
        st.info("Onglet LiftOver — Convertissez les coordonnées entre espèces.")

    # ═══════ T18 — Diagnostic ═══════
    with tabs[18]:
        if st.session_state.gt_filt is not None:
            render_diagnostic_dashboard(
                st.session_state.ind_filt, st.session_state.snp_filt,
                st.session_state.froh, st.session_state.roh_df,
                st.session_state.gwas_res, st.session_state.ne_dict, st.session_state.fst)

if __name__ == "__main__":
    main()
