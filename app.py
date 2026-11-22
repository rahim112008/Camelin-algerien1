"""🧬 Universal Livestock Genomics Pipeline v23.9 (Bovin & Camelin) — CORRIGÉ
Partie 1/4 : imports, config, utilitaires, parsers, QC.
"""
import gzip, io, warnings, zipfile, gc, hashlib, json, time
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

warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(page_title="🧬 Universal Genomics v23.9", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════════════════
# SPECIES_CONFIG — [FIX] MT en Mb, NBD→VIANDE, ARVANA→DROMADAIRE
# ══════════════════════════════════════════════════════════════════════════════
SPECIES_CONFIG = {
    "Camelin (Dromadaire)": {
        "taxon": "9838", "vep_species": "camelus_dromedarius", "prefix": "camel",
        "autosomes": [str(i) for i in range(1, 37)],
        "ars": {
            "1": 124.7, "2": 123.8, "3": 120.7, "4": 68.3, "5": 97.5, "6": 97.5,
            "7": 91.2, "8": 89.4, "9": 86.7, "10": 84.1, "11": 81.5, "12": 78.9,
            "13": 76.3, "14": 73.8, "15": 71.2, "16": 68.6, "17": 66.1, "18": 63.5,
            "19": 60.9, "20": 58.4, "21": 55.8, "22": 53.2, "23": 50.7, "24": 48.1,
            "25": 45.5, "26": 43.0, "27": 40.4, "28": 37.8, "29": 35.3, "30": 32.7,
            "31": 30.1, "32": 27.6, "33": 25.0, "34": 22.4, "35": 19.9, "36": 17.3,
            "X": 140.0, "Y": 45.0, "MT": 0.0166,   # [FIX] Mb
        },
        "groupes": {
            "Production laitière": {
                "HAUTE_LAITIERE": ["MOJAHEEM", "MAGHATEER", "WADAH", "AWARIK"],
                "FAIBLE_LAITIERE": ["BIKANERI", "JAISALMERI"],
            },
            "Sous-espèce": {
                "DROMADAIRE": ["MOJAHEEM", "MAGHATEER", "WADAH", "BIKANERI", "ARVANA"],  # [FIX]
                "BACTRIANE": ["KAZAKH_BACTRIAN"],
            },
        },
    },
    "Bovin (Taurin/Zébu)": {
        "taxon": "9913", "vep_species": "bos_taurus", "prefix": "bovine",
        "autosomes": [str(i) for i in range(1, 30)],
        "ars": {
            "1": 158.5, "2": 136.2, "3": 121.0, "4": 120.0, "5": 120.1, "6": 117.8,
            "7": 110.7, "8": 113.4, "9": 105.7, "10": 103.3, "11": 107.3, "12": 91.2,
            "13": 84.2, "14": 84.6, "15": 85.2, "16": 81.7, "17": 75.2, "18": 66.1,
            "19": 64.1, "20": 72.0, "21": 71.6, "22": 61.4, "23": 52.5, "24": 62.4,
            "25": 43.0, "26": 51.7, "27": 46.8, "28": 46.3, "29": 51.3,
            "X": 148.8, "Y": 57.8, "MT": 0.0163,
        },
        "groupes": {
            "Production": {
                "LAIT": ["NAR", "PTP", "PRS"],
                "VIANDE": ["LJR", "SHO", "YBS", "NBD"],  # [FIX]
            },
            "Sous-espèce": {
                "ZEBU": ["PTP", "PRS"],
                "TAURUS": ["LJR", "SHO", "NAR", "NBD"],
            },
        },
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# HACHAGE / CACHE
# ══════════════════════════════════════════════════════════════════════════════
def make_group_map(group_dict):
    return {r.upper(): grp for grp, races in group_dict.items() for r in races}

def apply_group_mapping(ind_df, group_map, default="AUTRE"):
    df = ind_df.copy()
    df["FID"] = df["FID"].astype(str).str.upper()
    df["FID_GROUPE"] = df["FID"].map(group_map).fillna(default)
    return df

def _ha(x):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return "empty"
    if x.dtype.kind in ("U", "S", "O"):
        h = '|'.join(map(str, x.ravel())).encode('utf-8', 'replace')
        return f"a|{x.shape}|{hashlib.blake2b(h, digest_size=16).hexdigest()}"
    return f"{x.shape}|{x.dtype}|{hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=16).hexdigest()}"

def _hn(x):
    try:
        return _ha(np.asarray(x))
    except Exception:
        return f"{type(x).__name__}|{len(x) if hasattr(x, '__len__') else 0}"

HF = {np.ndarray: _ha, pd.Series: _hn, pd.Index: _hn}
# [FIX] compatibilité pandas (pd.core.arrays est privé)
try:
    from pandas.api.extensions import ExtensionArray as _EA
    for _cls in vars(pd.arrays).values():
        if isinstance(_cls, type) and issubclass(_cls, _EA):
            HF[_cls] = _hn
except Exception:
    for _n in ("ArrowStringArray", "StringArray", "IntegerArray", "FloatingArray",
               "BooleanArray", "NumpyExtensionArray", "PandasArray",
               "DatetimeArray", "TimedeltaArray", "PeriodArray", "IntervalArray",
               "Categorical"):
        _cls = getattr(pd.arrays, _n, None)
        if _cls is not None:
            HF[_cls] = _hn

def cache_data(func=None, **kw):
    def _d(f):
        return st.cache_data(show_spinner=False, hash_funcs=HF, **kw)(f)
    return _d if func is None else _d(func)

# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES GÉNOMIQUES
# ══════════════════════════════════════════════════════════════════════════════
def impute_mean(gt):
    g = gt.astype(np.float32, copy=True)
    cm = np.nanmean(g, axis=0)
    cm = np.where(np.isnan(cm), 0.0, cm)
    nm = np.isnan(g)
    if nm.any():
        g[nm] = np.take(cm, np.where(nm)[1])
    return g

def _std_geno(gt):
    X = impute_mean(gt)
    X = X - X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-8] = 1.0
    return X / sd

def _autw(ars, autosomes):
    L = np.array([ars[c] for c in autosomes], dtype=float)
    return L / L.sum()

def _chk(c):
    s = str(c).upper().replace("CHROMOSOME", "").replace("CHR", "")
    if s.isdigit():
        return (0, int(s), "")
    sp = {"X": 100, "Y": 101, "MT": 102, "M": 102}
    return (1, sp[s], "") if s in sp else (2, 0, s)

def _chk_num(c):
    return _chk(c)[0] * 1000 + _chk(c)[1]

def sort_chr_bp(df, bp="BP"):
    return (df.assign(_k=df["CHR"].map(_chk_num), _s=df["CHR"].astype(str))
              .sort_values(["_k", "_s", bp], kind="stable")
              .drop(columns=["_k", "_s"]).reset_index(drop=True))

def _norm_chr(s):
    return s.astype(str).str.strip().str.replace(r"^(?i:chr)", "", regex=True)

def _rt(raw):
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw).decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")

def sample_fid_iid(s, fp=2):
    n = str(s).strip()
    for e in (".CEL", ".cel", ".TXT", ".txt", ".gz", ".GZ"):
        if n.endswith(e):
            n = n[:-len(e)]
    n = (n.replace("(", "_").replace(")", "").replace("[", "_").replace("]", "_")
          .replace(" ", "_").replace("/", "_"))
    while "__" in n:
        n = n.replace("__", "_")
    p = n.strip("_").split("_")
    return ("_".join(p[:fp]) if len(p) >= fp else p[0]), n

def _ca(a):
    a = str(a).strip() if a is not None else "0"
    return "0" if a in ("", ".", "-", "NA", "nan", "None", "0", "00") else a

def _sample_blocks(n, max_snp, block=200, seed=42):
    if n <= max_snp:
        return np.arange(n)
    nb = int(np.ceil(n / block))
    k = max(1, max_snp // block)
    ch = np.sort(np.random.default_rng(seed).choice(nb, size=min(k, nb), replace=False))
    return np.concatenate([np.arange(b * block, min((b + 1) * block, n)) for b in ch])

def _split_line(line):
    if "\t" in line:
        return [x.strip() for x in line.split("\t") if x.strip()]
    if "," in line and line.count(",") > line.count(" "):
        return [x.strip() for x in line.split(",") if x.strip()]
    return line.split()

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
def diagnose_ped(raw_bytes, fn=""):
    try:
        text = _rt(raw_bytes)
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return {"error": "Fichier vide"}
        sample = lines[:20]
        first = _split_line(sample[0])
        is_header = False
        if first:
            f0 = first[0].lower()
            if f0 in ("fid", "sample", "sample_id", "sampleid", "iid", "animal",
                      "id", "family", "individual"):
                is_header = True
        data_lines = lines[1:] if is_header else lines
        data_first = _split_line(data_lines[0])
        is_plink = False
        if len(data_first) >= 6:
            try:
                int(data_first[2]); int(data_first[3]); int(data_first[4])
                is_plink = True
            except (ValueError, IndexError):
                pass
        start = 6 if is_plink else 1
        n_snp = (len(data_first) - start) // 2
        if n_snp < 1:
            return {"error": f"Impossible d'extraire des génotypes : {len(data_first)} colonnes"}
        sep = ("tab" if "\t" in sample[0] else "comma" if "," in sample[0] else "space")
        return {"format": "PLINK standard" if is_plink else "non-standard (IID + génotypes)",
                "n_individuals": len(data_lines), "n_snps": n_snp, "separator": sep,
                "has_header": is_header,
                "first_id": data_first[1] if is_plink and len(data_first) > 1 else data_first[0],
                "first_genotypes": data_first[start:start + 6]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

def diagnose_map(raw_bytes, fn=""):
    try:
        text = _rt(raw_bytes)
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return {"error": "Fichier vide"}
        sample = lines[:20]
        first = _split_line(sample[0])
        is_header = False
        if first:
            f0 = first[0].lower().lstrip("#")
            if f0 in ("chr", "chrom", "chromosome", "snp", "marker", "id"):
                is_header = True
        data_lines = lines[1:] if is_header else lines
        data_first = _split_line(data_lines[0])
        is_standard = False
        if len(data_first) >= 4:
            try:
                float(data_first[2]); int(float(data_first[3]))
                is_standard = True
            except (ValueError, IndexError):
                pass
        sep = ("tab" if "\t" in sample[0] else "comma" if "," in sample[0] else "space")
        return {"format": "PLINK standard (CHR SNP CM BP)" if is_standard else "variante",
                "n_lines_data": len(data_lines), "n_columns": len(data_first),
                "separator": sep, "has_header": is_header,
                "first_line_tokens": data_first[:6]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

def preview_lines(raw_bytes, n=4):
    try:
        return [l for l in _rt(raw_bytes).splitlines() if l.strip()][:n]
    except Exception as e:
        return [f"❌ {e}"]

# ══════════════════════════════════════════════════════════════════════════════
# PARSEURS
# ══════════════════════════════════════════════════════════════════════════════
def parse_axiom(raw_bytes, fn=""):
    raw = gzip.decompress(raw_bytes) if fn.endswith(".gz") or raw_bytes[:2] == b"\x1f\x8b" else raw_bytes
    lines = [l.rstrip("\r\n") for l in raw.decode("utf-8", errors="replace").splitlines() if l.strip()]
    if not lines:
        raise ValueError("Fichier vide")
    start = 1 if (lines[0].startswith("#") or lines[0].startswith("Sample")
                  or "Sample Filename" in lines[0][:50]) else 0
    samples, gtypes = [], []
    for line in lines[start:]:
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            continue
        samples.append(parts[0].strip())
        alleles = []
        for cell in parts[1:]:
            tk = cell.split()
            if len(tk) == 2:
                alleles.append((_ca(tk[0]), _ca(tk[1])))
            elif len(tk) == 1 and len(tk[0]) == 2 and all(ch in "ACGT0" for ch in tk[0]):
                alleles.append((tk[0][0], tk[0][1]))
            else:
                alleles.append(("0", "0"))
        gtypes.append(alleles)
    if not samples:
        raise ValueError("Aucun échantillon")
    c = Counter([len(r) for r in gtypes]).most_common(1)[0][0]
    for i, r in enumerate(gtypes):
        gtypes[i] = r[:c] + [("0", "0")] * (c - len(r))
    return samples, gtypes, c

def parse_map(b):
    text = _rt(b)
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = _split_line(line)
        if not p or p[0].lower().lstrip("#") in ("chr", "chrom", "chromosome", "snp",
                                                 "marker", "id"):
            continue
        if len(p) >= 4:
            try:
                rows.append({"CHR": str(p[0]), "SNP": p[1],
                             "CM": float(p[2]) if p[2] not in ("0", ".", "NA", "") else 0.0,
                             "BP": int(float(p[3]))})
                continue
            except (ValueError, IndexError):
                pass
        if len(p) == 3:
            try:
                rows.append({"CHR": str(p[0]), "SNP": p[1], "CM": 0.0,
                             "BP": int(float(p[2]))})
                continue
            except (ValueError, IndexError):
                pass
        if len(p) == 3:
            try:
                bp = int(float(p[2]))
                if p[1].upper() in {str(i) for i in range(1, 40)} | {"X", "Y", "MT", "M"}:
                    rows.append({"CHR": str(p[1]), "SNP": p[0], "CM": 0.0, "BP": bp})
                    continue
            except (ValueError, IndexError):
                pass
        if len(p) >= 5:
            try:
                rows.append({"CHR": str(p[0]), "SNP": p[1], "CM": 0.0,
                             "BP": int(float(p[3]))})
                continue
            except (ValueError, IndexError):
                pass
    if not rows:
        raise ValueError(".map vide ou format non reconnu")
    df = pd.DataFrame(rows)
    df["CHR"] = _norm_chr(df["CHR"])
    if df["BP"].max() < 10000:
        df["BP"] = df["BP"] * 1_000_000
    return df

def parse_ped(b, n_map=None):
    text = _rt(b)
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError(".ped vide")
    first = _split_line(lines[0])
    is_header = bool(first) and first[0].lower() in (
        "fid", "sample", "sample_id", "sampleid", "iid", "animal", "id", "family", "individual"
    )
    if is_header:
        lines = lines[1:]
        if not lines:
            raise ValueError(".ped vide après en-tête")
    first_data = _split_line(lines[0])
    is_plink = False
    if len(first_data) >= 6:
        try:
            int(first_data[2]); int(first_data[3]); int(first_data[4])
            is_plink = True
        except (ValueError, IndexError):
            pass
    if is_plink:
        n_ped = (len(first_data) - 6) // 2
        start_idx, id_idx, fid_col = 6, 1, 0
    else:
        n_ped = (len(first_data) - 1) // 2
        start_idx, id_idx, fid_col = 1, 0, None
    if n_ped < 1:
        raise ValueError(f"Impossible d'extraire des génotypes : {len(first_data)} colonnes")
    n = n_ped if n_map is None else min(n_map, n_ped)
    ec = start_idx + 2 * n
    fids, iids, allele_to_idx, gt_rows = [], [], {}, []
    for line in lines:
        p = _split_line(line)
        if len(p) < ec:
            p = p + ["0"] * (ec - len(p))
        fids.append(p[fid_col] if fid_col is not None else "FAM")
        iids.append(p[id_idx])
        row = np.full(2 * n, -1, dtype=np.int32)
        for k, a in enumerate(p[start_idx:start_idx + 2 * n]):
            if a not in ("0", ".", "NA", "", "N", "N/N"):
                if a not in allele_to_idx:
                    allele_to_idx[a] = len(allele_to_idx)
                row[k] = allele_to_idx[a]
        gt_rows.append(row)
    if not iids:
        raise ValueError("Aucun individu valide")
    mat = np.stack(gt_rows)
    del gt_rows
    a1, a2 = mat[:, 0::2], mat[:, 1::2]
    miss = (a1 == -1) | (a2 == -1)
    ok = ~miss
    K = len(allele_to_idx)
    if K == 0:
        raise ValueError("Aucun allèle valide")
    cnt = np.zeros((K, n), dtype=np.int32)
    for k in range(K):
        cnt[k] = ((a1 == k) & ok).sum(0) + ((a2 == k) & ok).sum(0)
    idx_to_allele = {v: k for k, v in allele_to_idx.items()}
    order = np.argsort(-cnt, axis=0, kind="stable")
    major, minor = order[0], (order[1] if K > 1 else order[0])
    A1 = np.array([idx_to_allele[major[j]] if cnt[major[j], j] > 0 else "."
                   for j in range(n)])
    A2 = np.array([idx_to_allele[minor[j]] if cnt[minor[j], j] > 0 else "."
                   for j in range(n)])
    gt = ((a1 == minor[None, :]).astype(np.float32)
          + (a2 == minor[None, :]).astype(np.float32))
    gt[miss] = np.nan
    return gt, pd.DataFrame({"FID": fids, "IID": iids}), A1.tolist(), A2.tolist()

def parse_vcf(b):
    samples, rows, gts_list = [], [], []
    for line in _rt(b).splitlines():
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
        elif line.strip() and not line.startswith("#"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 10 and "," not in p[4]:
                fmt = p[8].split(":")
                if "GT" not in fmt:
                    continue
                gi = fmt.index("GT")
                gts = np.full(len(samples), np.nan, dtype=np.float32)
                for k, s in enumerate(p[9:9 + len(samples)]):
                    try:
                        al = s.split(":")[gi].replace("|", "/").split("/")
                        if not any(a in (".", "") for a in al):
                            gts[k] = sum(int(a) for a in al)
                    except (ValueError, IndexError):
                        pass
                try:
                    bp = int(float(p[1]))
                except (ValueError, TypeError):
                    continue
                rows.append({"CHR": str(p[0]),
                             "SNP": p[2] if p[2] not in ("", ".") else f"{p[0]}:{p[1]}",
                             "CM": 0.0, "BP": bp, "A1": p[3], "A2": p[4]})
                gts_list.append(gts)
    if not rows:
        raise ValueError("Aucun variant VCF")
    gt = np.column_stack(gts_list).astype(np.float32)
    for j in range(gt.shape[1]):
        v = gt[:, j][~np.isnan(gt[:, j])]
        if len(v) > 0 and v.mean() / 2.0 > 0.5:
            gt[:, j] = 2.0 - gt[:, j]
            rows[j]["A1"], rows[j]["A2"] = rows[j]["A2"], rows[j]["A1"]
    df_snp = pd.DataFrame(rows)
    df_snp["CHR"] = _norm_chr(df_snp["CHR"])
    return gt, pd.DataFrame({"FID": samples, "IID": samples}), df_snp

def _gff_attr(a, key):
    for part in a.split(";"):
        part = part.strip()
        if part.startswith(key + "="):
            return part[len(key) + 1:]
    return None

def parse_gff3(b):
    genes = []
    for line in _rt(b).splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split("\t")
        if len(p) < 9 or p[2] not in ("gene", "ncRNA_gene", "pseudogene"):
            continue
        a = p[8]
        name = None
        if 'gene_name "' in a:
            name = a.split('gene_name "')[1].split('"')[0]
        elif 'gene_id "' in a:
            name = a.split('gene_id "')[1].split('"')[0]
        else:
            for k in ("Name", "gene", "gene_name", "ID"):
                name = _gff_attr(a, k)
                if name:
                    break
        if not name:
            name = "Unknown"
        for pre in ("gene:", "gene-"):
            if name.startswith(pre):
                name = name[len(pre):]
        chrom = p[0][3:] if p[0].lower().startswith("chr") else p[0]
        try:
            genes.append({"CHR": str(chrom), "START": int(p[3]),
                          "END": int(p[4]), "GENE": name})
        except ValueError:
            continue
    return (pd.DataFrame(genes, columns=["CHR", "START", "END", "GENE"])
            .drop_duplicates().reset_index(drop=True))

@st.cache_data(show_spinner=False)
def load_gff(b):
    return parse_gff3(b)

def parse_gmt(file_bytes):
    sets = {}
    for line in file_bytes.decode("utf-8", errors="replace").splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 3 and [g for g in parts[2:] if g]:
            sets[parts[0]] = [g for g in parts[2:] if g]
    return sets

def load_pheno(b, fn=""):
    raw = gzip.decompress(b) if fn.endswith(".gz") else b
    text = raw.decode("utf-8", errors="replace")
    sep = "\t" if text.count("\t") > text.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python")

def match_pheno(ind_df, pheno_df, mapping_df=None):
    ind = ind_df.copy()
    ind["_k"] = ind["IID"].astype(str)
    if mapping_df is not None:
        p, m = pheno_df.copy(), mapping_df.copy()
        id_c = next((c for c in ["Animal", "IID", "id", "ID"] if c in p.columns), None)
        if id_c is None:
            raise ValueError("Colonne ID absente.")
        if id_c != "IID":
            p = p.drop(columns=["IID"], errors="ignore")
        p[id_c] = p[id_c].astype(str)
        map_a = next((c for c in m.columns if c != "IID"), None)
        if map_a is None or "IID" not in m.columns:
            raise ValueError("Mapping invalide.")
        m[map_a] = m[map_a].astype(str)
        m["IID"] = m["IID"].astype(str)
        pm = p.merge(m, left_on=id_c, right_on=map_a, how="left")
        pm["_k"] = pm["IID"].astype(str)
        pm = pm.drop_duplicates("_k")
        out = ind[["_k", "FID", "IID"]].merge(
            pm.drop(columns=["IID", "FID"], errors="ignore"), on="_k", how="left")
        return out.drop(columns="_k"), id_c
    id_c = next((c for c in ["IID", "id", "ID", "sample", "Animal", "animal"]
                 if c in pheno_df.columns), None)
    if id_c is None:
        raise ValueError("Aucune colonne ID.")
    p = pheno_df.copy()
    p["_k"] = p[id_c].astype(str)
    p = p.drop_duplicates("_k")
    out = ind[["_k", "FID", "IID"]].merge(
        p.drop(columns=[id_c, "FID"], errors="ignore"), on="_k", how="left")
    return out.drop(columns="_k"), id_c

# ══════════════════════════════════════════════════════════════════════════════
# DÉMO / UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════
def gen_demo(n_ind=150, n_snp=800, n_pop=4, seed=42, autosomes=None, ars=None):
    rng = np.random.default_rng(seed)
    if not autosomes or not ars:
        autosomes = [str(i) for i in range(1, 30)]
        ars = {c: 100.0 for c in autosomes}
    pops = [f"POP_{i}" for i in range(1, min(n_pop, 10) + 1)]
    pa = rng.beta(0.6, 0.6, n_snp)
    gt = np.zeros((n_ind, n_snp), dtype=np.float32)
    ir = []
    k = 0
    for pi_, p in enumerate(pops):
        pk = np.clip(pa + rng.normal(0, 0.15, n_snp), 0.02, 0.98)
        for i in range(n_ind // len(pops) + (1 if pi_ < n_ind % len(pops) else 0)):
            gt[k] = ((rng.random(n_snp) < pk).astype(np.int8)
                     + (rng.random(n_snp) < pk).astype(np.int8))
            ir.append({"FID": p, "IID": f"{p}_{i + 1:03d}"})
            k += 1
    gt[rng.random(gt.shape) < 0.02] = np.nan
    ch = rng.choice(autosomes, n_snp, p=_autw(ars, autosomes))
    bp = np.array([rng.integers(1, int(ars[c] * 1_000_000)) for c in ch], dtype=np.int64)
    snp = pd.DataFrame({"CHR": ch.astype(str),
                        "SNP": [f"rs{i:07d}" for i in range(n_snp)],
                        "CM": bp / 1e6, "BP": bp, "A1": "A", "A2": "G"})
    order = np.lexsort((bp, snp["CHR"].map(_chk_num).values))
    return gt[:, order], pd.DataFrame(ir), snp.iloc[order].reset_index(drop=True)

def rec_fid(ind):
    df = ind.copy()
    if df["FID"].nunique() <= 1:
        df["FID"] = df["IID"].astype(str).str.split("_").str[0]
    return df

def clean_ped_to_plink(raw_bytes, n_max_snp=None):
    gt, ind, a1, a2 = parse_ped(raw_bytes, n_max_snp)
    n, m = gt.shape
    lines = []
    for i in range(n):
        geno = []
        for j in range(m):
            v = gt[i, j]
            if np.isnan(v):
                geno += ["0", "0"]
            elif v == 0:
                geno += [a1[j], a1[j]]
            elif v == 1:
                geno += [a1[j], a2[j]]
            else:
                geno += [a2[j], a2[j]]
        lines.append("\t".join([str(ind.iloc[i]["FID"]), str(ind.iloc[i]["IID"]),
                                "0", "0", "0", "-9"] + geno))
    return "\n".join(lines) + "\n"

# ══════════════════════════════════════════════════════════════════════════════
# QC
# ══════════════════════════════════════════════════════════════════════════════
def miss_ind(g): return np.isnan(g).mean(axis=1)
def miss_snp(g): return np.isnan(g).mean(axis=0)
def maf(g): return np.minimum(np.nanmean(g, axis=0) / 2.0, 1.0 - np.nanmean(g, axis=0) / 2.0)

def het(g):
    ok = ~np.isnan(g)
    return np.where(ok.sum(1) > 0, ((g == 1) & ok).sum(1) / np.maximum(ok.sum(1), 1), np.nan)

def hwe_exact(nh, n0, n2):
    n = nh + n0 + n2
    if n == 0 or (nh == 0 and (n0 == 0 or n2 == 0)):
        return 1.0
    rare = 2 * min(n0, n2) + nh
    mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2:
        mid += 1
    pr = np.zeros(rare + 1)
    pr[mid] = 1.0
    s = 1.0
    ch, cr, cc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch <= rare - 2:
        pr[ch + 2] = pr[ch] * 4 * cr * cc / ((ch + 2) * (ch + 1))
        s += pr[ch + 2]; ch += 2; cr -= 1; cc -= 1
    ch, cr, cc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch >= 2:
        pr[ch - 2] = pr[ch] * ch * (ch - 1) / (4 * (cr + 1) * (cc + 1))
        s += pr[ch - 2]; ch -= 2; cr += 1; cc += 1
    po = pr[nh] if nh < len(pr) else 0.0
    return float(min(pr[pr <= po + 1e-7].sum() / s, 1.0))

@cache_data
def hwe_pvals(g):
    pv = np.full(g.shape[1], np.nan)
    n0 = ((g == 0) & ~np.isnan(g)).sum(0)
    n1 = ((g == 1) & ~np.isnan(g)).sum(0)
    n2 = ((g == 2) & ~np.isnan(g)).sum(0)
    nv = n0 + n1 + n2
    if g.shape[1] <= 20000:
        for j in range(g.shape[1]):
            if nv[j] >= 5:
                pv[j] = hwe_exact(int(n1[j]), int(n0[j]), int(n2[j]))
    else:
        p = (n1 + 2 * n2) / (2.0 * np.where(nv > 0, nv, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            eh = 2.0 * p * (1.0 - p) * nv
            ok = (eh >= 5) & np.isfinite(eh)
            chi2 = (np.abs(n1 - eh) - 0.5) ** 2 / eh
            pv[ok] = stats.chi2.sf(chi2[ok], df=1)
    return pv

# [FIX] fonction manquante dans l'original : HWE par population
@cache_data
def hwe_by_pop(gt, labels, min_n=10):
    labels = np.asarray(labels).astype(str)
    pops = [p for p in np.unique(labels) if (labels == p).sum() >= min_n]
    if not pops:
        return pd.DataFrame()
    cols = {p: hwe_pvals(gt[labels == p]) for p in pops}
    return pd.DataFrame(cols)

def sync_dims(gt, ind_df, snp_df):
    n = min(gt.shape[0], len(ind_df))
    m = min(gt.shape[1], len(snp_df))
    return gt[:n, :m], ind_df.iloc[:n].reset_index(drop=True), snp_df.iloc[:m].reset_index(drop=True)

def apply_qc(gt, ind_df, snp_df, params, autosomes, keep_autosomes=True, rem_out=True):
    gt, ind_df, snp_df = sync_dims(gt, ind_df, snp_df)
    n0, m0 = gt.shape
    tr = {}
    if keep_autosomes:
        auto = {str(a).upper() for a in autosomes}
        mask = (snp_df["CHR"].astype(str).str.upper()
                .str.replace(r"^CHR", "", regex=True).isin(auto).values)
        gt = gt[:, mask]
        snp_df = snp_df[mask].reset_index(drop=True)
        tr["autosomes"] = int(m0 - gt.shape[1])
    m = gt.shape[1]
    k = miss_snp(gt) <= params["geno"]
    gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True)
    tr["geno"] = int(m - gt.shape[1]); m = gt.shape[1]
    k = miss_ind(gt) <= params["mind"]
    gt = gt[k]; ind_df = ind_df[k].reset_index(drop=True)
    tr["mind"] = int(n0 - gt.shape[0])
    mf = maf(gt)
    k = np.isfinite(mf) & (mf >= params["maf"])
    gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True)
    tr["maf"] = int(m - gt.shape[1]); m = gt.shape[1]
    if gt.shape[1] == 0:
        raise ValueError("Tous les SNPs exclus (MAF)")
    if params.get("hwe_pop"):
        pvd = hwe_by_pop(gt, ind_df["FID"].astype(str).values,
                         int(params.get("hwe_min_n", 10)))
        if pvd.shape[1] == 0:
            raise ValueError("Aucune population avec assez d'individus pour le HWE")
        k = (pvd.values < params["hwe"]).sum(1) < int(params.get("hwe_k", 1))
        tr["hwe_pops_testées"] = int(pvd.shape[1])
    else:
        pv = hwe_pvals(gt)
        k = np.isnan(pv) | (pv >= params["hwe"])
    gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True)
    tr["hwe"] = int(m - gt.shape[1])
    if gt.shape[1] == 0:
        raise ValueError("Tous les SNPs exclus (HWE)")
    n_pre = int(gt.shape[0])
    if rem_out:
        h = het(gt)
        kp = np.ones(len(h), dtype=bool)
        for f in ind_df["FID"].unique():
            msk = (ind_df["FID"] == f).values
            sub = h[msk]
            if len(sub) >= 5 and np.nanstd(sub) > 1e-9:
                z = (sub - np.nanmean(sub)) / np.nanstd(sub)
                idx = np.where(msk)[0]
                kp[idx[np.abs(z) > params["het_sd"]]] = False
        gt = gt[kp]; ind_df = ind_df[kp].reset_index(drop=True)
        tr["het"] = int(n_pre - len(ind_df))
    gc.collect()
    return gt, ind_df, snp_df, {
        "n_ind_init": int(n0), "n_snp_init": int(m0), "n_ind_pre": n_pre,
        "n_ind_final": int(gt.shape[0]), "n_snp_final": int(gt.shape[1]),
        "excluded_ind": int(n0 - gt.shape[0]), "excluded_snp": int(m0 - gt.shape[1]),
        "trace": tr,
    }

# ══════════════════════════════════════════════════════════════════════════════
# Fst (Weir & Cockerham 1984)
# ══════════════════════════════════════════════════════════════════════════════
def _wc_abc(gt, labels, min_n=2):
    labels = np.asarray(labels).astype(str)
    pops = np.unique(labels)
    m = gt.shape[1]
    n = np.zeros((len(pops), m))
    p = np.zeros_like(n)
    h = np.zeros_like(n)
    for k, pop in enumerate(pops):
        g = gt[labels == pop]
        ok = ~np.isnan(g)
        nn = ok.sum(0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            pk = np.where(nn > 0, np.nansum(g, 0) / (2 * nn), 0.0)
            hk = np.where(nn > 0, ((g == 1) & ok).sum(0) / nn, 0.0)
        use = nn >= min_n
        n[k] = np.where(use, nn, 0.0)
        p[k] = np.where(use, pk, 0.0)
        h[k] = np.where(use, hk, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (n > 0).sum(0).astype(float)
        nsum = n.sum(0)
        nbar = nsum / r
        pbar = (n * p).sum(0) / nsum
        s2 = (n * (p - pbar) ** 2).sum(0) / (nbar * (r - 1))
        hbar = (n * h).sum(0) / nsum
        nc = (r * nbar - (n ** 2).sum(0) / (r * nbar)) / (r - 1)
        a = (nbar / nc) * (s2 - (1.0 / (nbar - 1))
                            * (pbar * (1 - pbar) - (r - 1) / r * s2 - hbar / 4))
        b = (nbar / (nbar - 1)) * (pbar * (1 - pbar) - (r - 1) / r * s2
                                    - (2 * nbar - 1) / (4 * nbar) * hbar)
        den = a + b + hbar / 2.0
    valid = (r >= 2) & (nbar > 1) & (nc > 0) & np.isfinite(den) & np.isfinite(a)
    return np.where(valid, a, np.nan), np.where(valid, b, np.nan), np.where(valid, hbar / 2.0, np.nan)

def _wc_components(gt, labels, min_n=2):
    a, b, c = _wc_abc(gt, labels, min_n)
    return a, a + b + c

@cache_data
def fst_snp_wc(gt, labels):
    a, den = _wc_components(gt, labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = a / den
    return np.where(np.isfinite(f) & (den > 1e-9), np.clip(f, 0.0, None), np.nan)

def fst_global_wc(gt, labels):
    a, den = _wc_components(gt, labels)
    ok = np.isfinite(a) & np.isfinite(den)
    return (float(max(0.0, a[ok].sum() / den[ok].sum()))
            if ok.any() and den[ok].sum() > 0 else np.nan)

@cache_data
def fst_pairwise(gt, labels, use_wc84=True):
    labels = np.asarray(labels).astype(str)
    pops = sorted(np.unique(labels))
    K = len(pops)
    M = np.full((K, K), np.nan)
    np.fill_diagonal(M, 0.0)
    for i in range(K):
        for j in range(i + 1, K):
            mi, mj = labels == pops[i], labels == pops[j]
            if use_wc84:
                v = fst_global_wc(gt[mi | mj], labels[mi | mj])
            else:
                g1, g2 = gt[mi], gt[mj]
                p1 = np.nanmean(g1, 0) / 2.0
                p2 = np.nanmean(g2, 0) / 2.0
                n1 = (~np.isnan(g1)).sum(0)
                n2 = (~np.isnan(g2)).sum(0)
                dn = np.where(n1 + n2 > 0, n1 + n2, np.nan)
                pb = (p1 * n1 + p2 * n2) / dn
                hs = (2 * p1 * (1 - p1) * n1 + 2 * p2 * (1 - p2) * n2) / dn
                ht = 2 * pb * (1 - pb)
                ok = np.isfinite(ht) & np.isfinite(hs)
                v = (float(max(0.0, (ht[ok] - hs[ok]).sum() / ht[ok].sum()))
                     if ok.any() and ht[ok].sum() > 0 else np.nan)
            M[i, j] = M[j, i] = v
    return M, pops

@cache_data
def windowed_fst(gt, snp_df, labels, w=50, step=25):
    a, den = _wc_components(gt, labels)
    snp = snp_df.reset_index(drop=True).copy()
    n = min(len(snp), len(a))
    snp = snp.iloc[:n].copy()
    a = a[:n]; den = den[:n]
    snp["_a"] = a; snp["_d"] = den
    with np.errstate(invalid="ignore", divide="ignore"):
        snp["FST"] = np.where(np.isfinite(den) & (den > 1e-9),
                              np.clip(a / den, 0.0, None), np.nan)
    res = []
    for c in snp["CHR"].astype(str).unique():
        sub = snp[snp["CHR"].astype(str) == c].sort_values("BP").reset_index(drop=True)
        if len(sub) < w:
            continue
        A, Dn, F, B, S = (sub["_a"].values, sub["_d"].values, sub["FST"].values,
                          sub["BP"].values, sub["SNP"].values)
        for i in range(0, len(sub) - w + 1, step):
            sl = slice(i, i + w)
            okm = np.isfinite(A[sl]) & np.isfinite(Dn[sl])
            if okm.sum() < 5:
                continue
            dd = Dn[sl][okm].sum()
            if dd <= 1e-9:
                continue
            f_sl = F[sl]
            top = int(np.nanargmax(f_sl)) if np.isfinite(f_sl).any() else 0
            res.append({"CHR": c, "START": int(B[sl].min()), "END": int(B[sl].max()),
                        "MID": int((B[sl].min() + B[sl].max()) / 2), "N_SNP": w,
                        "MEAN_FST": float(max(0.0, A[sl][okm].sum() / dd)),
                        "MAX_FST": float(np.nanmax(f_sl)) if np.isfinite(f_sl).any() else np.nan,
                        "TOP_SNP": S[sl][top]})
    return pd.DataFrame(res, columns=["CHR", "START", "END", "MID", "N_SNP",
                                      "MEAN_FST", "MAX_FST", "TOP_SNP"])

@cache_data
def detect_outliers(win_df, z_thr=3.0):
    df = win_df.copy()
    if df.empty:
        df["Z_FST"] = []; df["OUTLIER"] = []
        return df
    mu, sd = df["MEAN_FST"].mean(), df["MEAN_FST"].std()
    df["Z_FST"] = (df["MEAN_FST"] - mu) / sd if sd > 1e-9 else 0.0
    df["OUTLIER"] = df["Z_FST"] >= z_thr
    return df.sort_values("MEAN_FST", ascending=False).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# PCA / MDS / diversité / Shannon
# ══════════════════════════════════════════════════════════════════════════════
@cache_data
def pca_analysis(gt, nc=10):
    X = impute_mean(gt)
    X = X - X.mean(0)
    n = min(nc, X.shape[0] - 1, X.shape[1])
    p = PCA(n_components=max(1, n))
    return p.fit_transform(X), p.explained_variance_ratio_ * 100.0

@cache_data
def mds_analysis(gt, nc=5):
    X = impute_mean(gt)
    n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i] = np.abs(X - X[i]).sum(1) / X.shape[1]
    kw = dict(n_components=min(nc, n - 1), random_state=42, n_init=1, max_iter=300)
    try:
        return SklearnMDS(dissimilarity="precomputed", normalized_stress=False, **kw).fit_transform(D)
    except Exception:
        return SklearnMDS(metric="precomputed", **kw).fit_transform(D)

@cache_data
def compute_diversity(gt, labels):
    labels = np.asarray(labels).astype(str)
    rows = []
    for p in np.unique(labels):
        sub = gt[labels == p]
        if sub.shape[0] < 3:
            continue
        ok = ~np.isnan(sub)
        n = ok.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            freq = np.where(n > 0, np.nansum(sub, 0) / (2 * n), np.nan)
            ho_snp = np.where(n > 0, ((sub == 1) & ok).sum(0) / n, np.nan)
        v = np.isfinite(freq)
        if not v.any():
            continue
        f = freq[v]; q = 1 - f
        he = float(np.mean(2 * f * q))
        ho = float(np.nanmean(ho_snp[v]))
        pic = float(np.mean(1 - (f ** 2 + q ** 2) - 2 * (f ** 2) * (q ** 2)))
        rows.append({"Population": p, "N": int(sub.shape[0]),
                     "H_Observee (Ho)": round(ho, 4),
                     "H_Attendue (He)": round(he, 4),
                     "Fis (Consanguinité)": round(1 - ho / he, 4) if he > 0 else 0.0,
                     "PIC": round(pic, 4),
                     "MAF_moyenne": round(float(np.mean(np.minimum(f, q))), 4),
                     "SNPs_Polymorphes": int(((f > 0) & (f < 1)).sum())})
    return pd.DataFrame(rows)

@cache_data
def shannon_weaver(gt, labels, min_n=3):
    labels = np.asarray(labels).astype(str)
    rows = []
    for pop in np.unique(labels):
        sub = gt[labels == pop]
        if sub.shape[0] < min_n:
            continue
        ok = ~np.isnan(sub)
        n = ok.sum(0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(n > 0, np.nansum(sub, 0) / (2.0 * n), np.nan)
        q = 1.0 - p
        with np.errstate(divide="ignore", invalid="ignore"):
            h_snp = -(np.where(p > 0, p * np.log(p), 0.0)
                      + np.where(q > 0, q * np.log(q), 0.0))
            j_snp = np.where(np.isfinite(h_snp), h_snp / np.log(2.0), np.nan)
        valid = np.isfinite(h_snp)
        if not valid.any():
            continue
        pv, qv = p[valid], q[valid]
        with np.errstate(invalid="ignore", divide="ignore"):
            ne_eff = 1.0 / np.clip(pv ** 2 + qv ** 2, 1e-12, 1.0)
        rows.append({"Population": pop, "N": int(sub.shape[0]),
                     "Shannon_H": round(float(np.nanmean(h_snp[valid])), 4),
                     "Shannon_H_SD": round(float(np.nanstd(h_snp[valid])), 4),
                     "Pielou_J": round(float(np.nanmean(j_snp[valid])), 4),
                     "Nb_Alleles_Efficaces": round(float(np.nanmean(ne_eff)), 4),
                     "Nb_SNPs_Polymorphes": int(valid.sum())})
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# KING / GRM / NMF / Admixture
# ══════════════════════════════════════════════════════════════════════════════
@cache_data
def king_mat(gt):
    ok = ~np.isnan(gt)
    H = ((gt == 1) & ok).astype(np.float32)
    G0 = ((gt == 0) & ok).astype(np.float32)
    G2 = ((gt == 2) & ok).astype(np.float32)
    h = H.sum(1).astype(np.float64)
    BH = H @ H.T
    ibs0 = G0 @ G2.T + G2 @ G0.T
    d = h[:, None] + h[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        K = (BH - 2 * ibs0) / d
    K[~np.isfinite(K)] = 0.0
    np.fill_diagonal(K, 0.5)
    return K

@cache_data
def grm_vanraden(gt):
    X = impute_mean(gt)
    p = X.mean(0) / 2.0
    Z = X - 2 * p
    den = 2 * np.sum(p * (1 - p))
    if den <= 0:
        return np.eye(gt.shape[0])
    return ((Z @ Z.T) / den).astype(np.float64)

@cache_data
def admixture_nmf(gt, K=3, seed=42):
    m = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=500)
    W = m.fit_transform(np.clip(impute_mean(gt), 0.0, 2.0))
    s = W.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return W / s, m.components_

@cache_data
def choose_best_k(gt, k_min=2, k_max=8, seed=42):
    X = np.clip(impute_mean(gt), 0.0, 2.0)
    rows = []
    for k in range(k_min, k_max + 1):
        m = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=400)
        W = m.fit_transform(X)
        rows.append({"K": k,
                     "RMSE": float(np.sqrt(np.mean((X - W @ m.components_) ** 2)))})
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# LD / Ne / ROH
# ══════════════════════════════════════════════════════════════════════════════
@cache_data
def ld_pruning(gt, chrs, w=50, step=5, r2=0.2):
    n = gt.shape[1]
    keep = np.ones(n, dtype=bool)
    if n < 3:
        return keep
    chrs = np.asarray(chrs).astype(str)
    X = _std_geno(gt).astype(np.float32)
    N = X.shape[0]
    for c in pd.unique(chrs):
        idx = np.where(chrs == c)[0]
        for s in range(0, len(idx), step):
            win = idx[s:s + w]
            win = win[keep[win]]
            if len(win) < 2:
                continue
            R2 = ((X[:, win].T @ X[:, win]) / N) ** 2
            for a in range(len(win) - 1):
                if not keep[win[a]]:
                    continue
                hits = np.where(R2[a, a + 1:] > r2)[0] + a + 1
                keep[win[hits]] = False
    return keep

def _pair_r2(X, bp, chrs, win=100, max_bp=None):
    n = X.shape[1]
    N = X.shape[0]
    D, R = [], []
    for i in range(n - 1):
        j1 = min(i + 1 + win, n)
        same = chrs[i + 1:j1] == chrs[i]
        if not same.any():
            continue
        d = np.abs(bp[i + 1:j1] - bp[i])
        ok = same & (d > 0)
        if max_bp is not None:
            ok &= (d <= max_bp)
        if ok.any():
            r = (X[:, i + 1:j1][:, ok].T @ X[:, i]) / N
            D.append(d[ok])
            R.append(r.astype(np.float64) ** 2)
    return (np.concatenate(D), np.concatenate(R)) if D else (np.array([]), np.array([]))

def _sorted_snps(gt, snp_df):
    s = snp_df.reset_index(drop=True).copy()
    s["CHR"] = s["CHR"].astype(str)
    s["BP"] = s["BP"].astype(float)
    order = np.lexsort((s["BP"].values, s["CHR"].map(_chk_num).values))
    return gt[:, order], s.iloc[order].reset_index(drop=True)

@cache_data
def ld_decay(gt, snp_df, max_kb=1000, max_snp=20000):
    if gt.shape[1] < 10:
        return pd.DataFrame(columns=["dist_kb", "r2", "count"])
    gt, s = _sorted_snps(gt, snp_df)
    idx = _sample_blocks(gt.shape[1], max_snp)
    gt = gt[:, idx]
    s = s.iloc[idx].reset_index(drop=True)
    X = _std_geno(gt)
    dist, r2 = _pair_r2(X, s["BP"].values, s["CHR"].values, 100, max_kb * 1000)
    if len(dist) == 0:
        return pd.DataFrame(columns=["dist_kb", "r2", "count"])
    df = pd.DataFrame({"dist_kb": dist / 1000.0, "r2": r2})
    df["bin"] = pd.cut(df["dist_kb"], bins=np.linspace(0, max_kb, 30),
                       labels=False, include_lowest=True)
    return (df.groupby("bin")
              .agg(dist_kb=("dist_kb", "mean"), r2=("r2", "mean"), count=("r2", "count"))
              .dropna().sort_values("dist_kb").reset_index(drop=True))

@cache_data
def ne_historical(gt, snp_df, max_snp=10000):
    N = gt.shape[0]
    if N < 5 or gt.shape[1] < 20:
        return pd.DataFrame({"GenAgo": [], "Ne": []})
    gt, s = _sorted_snps(gt, snp_df)
    idx = _sample_blocks(gt.shape[1], max_snp)
    gt = gt[:, idx]
    s = s.iloc[idx].reset_index(drop=True)
    X = _std_geno(gt)
    d, r2 = _pair_r2(X, s["BP"].values, s["CHR"].values, 50, 50e6)
    d = d / 1e6
    m = (d >= 0.001) & (d <= 50)
    if m.sum() < 20:
        return pd.DataFrame({"GenAgo": [], "Ne": []})
    d, r2 = d[m], r2[m]
    bins = np.logspace(np.log10(d.min()), np.log10(d.max()), 20)
    rows = []
    for k in range(len(bins) - 1):
        sel = (d >= bins[k]) & (d < bins[k + 1])
        if sel.sum() >= 10:
            rows.append((np.sqrt(bins[k] * bins[k + 1]), r2[sel].mean()))
    if len(rows) < 3:
        return pd.DataFrame({"GenAgo": [], "Ne": []})
    dm, rm = np.array(rows).T
    c = dm * 0.01
    r2a = np.clip(rm - 1.0 / N, 1e-6, 0.999)
    ne = (1.0 / r2a - 1.0) / (4.0 * c)
    t = 1.0 / (2.0 * c)
    df = (pd.DataFrame({"GenAgo": t, "Ne": ne})
            .replace([np.inf, -np.inf], np.nan).dropna())
    df = df[df["Ne"] > 0].sort_values("GenAgo").reset_index(drop=True)
    if len(df) > 3:
        df["Ne"] = df["Ne"].rolling(3, center=True, min_periods=1).mean()
    return df

@cache_data
def detect_roh(gt, chrs, bps, min_snps=30, min_kb=500.0):
    chrs = np.asarray(chrs).astype(str)
    bps = np.asarray(bps).astype(np.float64)
    if bps.max() < 1_000_000 and bps.max() > 0:
        bps = bps * 1_000_000
    bps = bps.astype(np.int64)
    n_ind = gt.shape[0]
    rohs = []
    fb = np.zeros(n_ind)
    tot = 0.0
    for c in pd.unique(chrs):
        idxs = np.where(chrs == c)[0]
        if len(idxs) < min_snps:
            continue
        idxs = idxs[np.argsort(bps[idxs], kind="stable")]
        pos = bps[idxs]
        sp = pos[-1] - pos[0]
        if sp < 1:
            continue
        tot += sp
        G = gt[:, idxs]
        for i in range(n_ind):
            bnd = np.concatenate(([-1], np.where((G[i] == 1) | np.isnan(G[i]))[0],
                                  [len(idxs)]))
            for a, b in zip(bnd[:-1], bnd[1:]):
                s_, e_ = a + 1, b - 1
                if e_ < s_:
                    continue
                nr = e_ - s_ + 1
                lkb = (pos[e_] - pos[s_]) / 1000.0
                if nr >= min_snps and lkb >= min_kb:
                    rohs.append({"IID_idx": i, "CHR": c, "start_bp": int(pos[s_]),
                                 "end_bp": int(pos[e_]), "n_snp": int(nr),
                                 "length_kb": float(lkb)})
                    fb[i] += pos[e_] - pos[s_]
    cols = ["IID_idx", "CHR", "start_bp", "end_bp", "n_snp", "length_kb"]
    return (pd.DataFrame(rohs, columns=cols),
            (fb / tot if tot > 0 else fb))

# ══════════════════════════════════════════════════════════════════════════════
# Tajima / composite score
# ══════════════════════════════════════════════════════════════════════════════
@cache_data
def tajimas_window(genos):
    ni, ns = genos.shape
    if ns < 3 or ni < 4:
        return np.nan, 0, np.nan, np.nan
    S = 0
    piv, thv = [], []
    for j in range(ns):
        col = genos[:, j]
        col = col[~np.isnan(col)]
        if len(col) < 2:
            continue
        p = col.mean() / 2.0
        N = 2 * len(col)
        if 0 < p < 1:
            S += 1
            piv.append(2 * p * (1 - p) * N / (N - 1))
            thv.append(1.0 / np.sum(1.0 / np.arange(1, N)))
    if S == 0 or not piv:
        return np.nan, int(S), np.nan, np.nan
    pi = float(np.sum(piv))
    th = float(np.sum(thv))
    n = ni * 2
    a1 = np.sum(1.0 / np.arange(1, n))
    a2 = np.sum(1.0 / (np.arange(1, n) ** 2))
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n ** 2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - 1 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / (a1 ** 2)
    e1 = c1 / a1
    e2 = c2 / (a1 ** 2 + a2)
    den = np.sqrt(e1 * S + e2 * S * (S - 1))
    if den <= 1e-12:
        return np.nan, int(S), pi, th
    return float((pi - S / a1) / den), int(S), pi, th

@cache_data
def tajimas_scan(gt, snp_df, w=25, step=10):
    snp = snp_df.copy().reset_index(drop=True)
    n = min(gt.shape[1], len(snp))
    snp = snp.iloc[:n].reset_index(drop=True)
    gt = gt[:, :n]
    res = []
    for c in snp["CHR"].unique():
        idx = np.where(snp["CHR"].values == c)[0]
        if len(idx) < w:
            continue
        idx = idx[np.argsort(snp["BP"].values[idx])]
        for i in range(0, len(idx) - w + 1, step):
            wi = idx[i:i + w]
            D, S, pis, ths = tajimas_window(gt[:, wi])
            bpw = snp["BP"].values[wi]
            span_kb = max((bpw.max() - bpw.min()) / 1000.0, 1e-9)
            if not np.isnan(D):
                res.append({"CHR": str(c), "START": int(bpw.min()),
                            "END": int(bpw.max()),
                            "MID": int((bpw.min() + bpw.max()) / 2),
                            "N_SNP": len(wi), "S": S, "TAJIMA_D": D,
                            "PI_PAR_SNP": pis / len(wi),
                            "THETA_W_PAR_SNP": ths / len(wi),
                            "PI_PAR_KB": pis / span_kb})
    return pd.DataFrame(res, columns=["CHR", "START", "END", "MID", "N_SNP", "S",
                                      "TAJIMA_D", "PI_PAR_SNP",
                                      "THETA_W_PAR_SNP", "PI_PAR_KB"])

@cache_data
def composite_score(gt, snp_df, labels):
    fst = fst_snp_wc(gt, labels)
    ok = ~np.isnan(gt)
    n = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        hom = np.where(n > 0, (((gt == 0) | (gt == 2)) & ok).sum(0) / n, np.nan)
        mv = np.nanmean(gt, axis=0) / 2.0
    mv = np.minimum(mv, 1 - mv)
    df = snp_df.copy().reset_index(drop=True)
    m = min(len(df), len(fst))
    df = df.iloc[:m].reset_index(drop=True)
    df["FST"] = fst[:m]; df["HOM"] = hom[:m]; df["MAF"] = mv[:m]
    def norm(x):
        mu, sd = np.nanmean(x), np.nanstd(x)
        return (x - mu) / sd if sd > 1e-9 else np.zeros_like(x)
    df["Z_FST"] = norm(df["FST"].values)
    df["Z_HOM"] = norm(df["HOM"].values)
    df["CSS"] = np.abs(df["Z_FST"] * df["Z_HOM"])
    return df.sort_values("CSS", ascending=False).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# Reynolds / UPGMA
# ══════════════════════════════════════════════════════════════════════════════
def _reynolds_D(g, labels, pops):
    ph = {p: np.nanmean(g[labels == p], 0) / 2.0 for p in pops}
    K = len(pops)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            pi_, pj = ph[pops[i]], ph[pops[j]]
            ok = np.isfinite(pi_) & np.isfinite(pj)
            num = np.sum((pi_[ok] - pj[ok]) ** 2)
            dn = np.sum(pi_[ok] * (1 - pj[ok]) + pj[ok] * (1 - pi_[ok]))
            if dn > 0:
                D[i, j] = D[j, i] = -np.log(1.0 - min(num / dn, 0.999999))
    return D

@cache_data
def reynolds(gt, labels):
    labels = np.asarray(labels).astype(str)
    pops = sorted(np.unique(labels))
    return _reynolds_D(gt, labels, pops), pops

@cache_data
def upgma_bootstrap(gt, snp_df, labels, n_boot=100, seed=42):
    labels = np.asarray(labels).astype(str)
    pops = sorted(np.unique(labels))
    K = len(pops)
    n_snp = gt.shape[1]
    if K < 2:
        raise ValueError("Au moins 2 populations requises")
    Do = _reynolds_D(gt, labels, pops)
    Zo = linkage(squareform(Do, checks=False), method="average")
    nl = K
    def clusters(Z):
        cl = {}
        for k, row in enumerate(Z):
            i, j = int(row[0]), int(row[1])
            l = frozenset([i]) if i < nl else cl[i]
            r = frozenset([j]) if j < nl else cl[j]
            cl[nl + k] = frozenset(l | r)
        return cl
    co = clusters(Zo)
    sc = np.zeros(len(Zo))
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.choice(n_snp, n_snp, replace=True)
        Zb = linkage(squareform(_reynolds_D(gt[:, idx], labels, pops), checks=False),
                     method="average")
        cb = set(clusters(Zb).values())
        for k, cl in co.items():
            if cl in cb:
                sc[k - nl] += 1
    supp = (sc / n_boot * 100).round(0)
    tr = to_tree(Zo)
    def nw(node):
        if node.is_leaf():
            return f"{pops[node.id]}"
        l = nw(node.get_left())
        r = nw(node.get_right())
        return (f"({l},{r}){int(supp[node.id - nl])}:{node.dist:.4f}"
                if node.id >= nl else f"({l},{r}):{node.dist:.4f}")
    return {"tree": tr, "Z": Zo, "newick": nw(tr) + ";",
            "support": supp, "D": Do, "pops": pops}

@cache_data
def midpoint_info(D, labels):
    D = np.asarray(D, dtype=float)
    n = len(labels)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    im, jm, md = 0, 1, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] > md:
                md = D[i, j]; im, jm = i, j
    return {"leaf_i": labels[im], "leaf_j": labels[jm],
            "max_dist": float(md), "midpoint": float(md / 2)}

# ══════════════════════════════════════════════════════════════════════════════
# GWAS (pairwise + LMM)
# ══════════════════════════════════════════════════════════════════════════════
def _logistic_irls(X, y, max_iter=25, tol=1e-6):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-9
        H = (X.T * W) @ X + 1e-8 * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return float(beta[-1]), None
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    W = mu * (1 - mu) + 1e-9
    try:
        cov = np.linalg.inv((X.T * W) @ X + 1e-8 * np.eye(X.shape[1]))
        se = float(np.sqrt(cov[-1, -1]))
    except np.linalg.LinAlgError:
        return float(beta[-1]), None
    return float(beta[-1]), se

def _pairwise_core(gt, snp_df, groups, ga, gb, maf_min=0.05,
                   method="allelic", covs=None):
    groups = np.asarray(groups).astype(str)
    mask = np.isin(groups, [ga, gb])
    if mask.sum() < 20:
        raise ValueError(f"<20 individus pour {ga} vs {gb}")
    g = gt[mask]
    y = (groups[mask] == ga).astype(int)
    if (y == 1).sum() < 5 or (y == 0).sum() < 5:
        raise ValueError("Moins de 5 individus dans une classe")
    snp = snp_df.reset_index(drop=True)
    n = min(g.shape[1], len(snp))
    g = g[:, :n]
    snp = snp.iloc[:n].reset_index(drop=True)
    cv = covs[mask] if covs is not None else None
    gA, gB = g[y == 1], g[y == 0]
    nA = (~np.isnan(gA)).sum(0).astype(float)
    nB = (~np.isnan(gB)).sum(0).astype(float)
    sA = np.nansum(gA, 0)
    sB = np.nansum(gB, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pA = sA / (2 * nA); pB = sB / (2 * nB)
        tot = (sA + sB) / (2 * (nA + nB))
        mj = np.minimum(tot, 1 - tot)
        hs = (2 * pA * (1 - pA) * nA + 2 * pB * (1 - pB) * nB) / (nA + nB)
        pb = (pA * nA + pB * nB) / (nA + nB)
        ht = 2 * pb * (1 - pb)
        fv = np.where(ht > 1e-9, np.clip((ht - hs) / ht, 0, None), 0.0)
    valid = (nA >= 5) & (nB >= 5) & np.isfinite(mj) & (mj >= maf_min)
    if method == "allelic":
        a1, a0, b1, b0 = sA, 2 * nA - sA, sB, 2 * nB - sB
        N = a1 + a0 + b1 + b0
        with np.errstate(invalid="ignore", divide="ignore"):
            den = (a1 + a0) * (b1 + b0) * (a1 + b1) * (a0 + b0)
            chi2 = N * (a1 * b0 - a0 * b1) ** 2 / den
        valid &= np.isfinite(chi2) & (den > 0)
        pv = np.where(valid, stats.chi2.sf(np.where(valid, chi2, 0), 1), np.nan)
    else:
        chi2 = np.full(n, np.nan); pv = np.full(n, np.nan)
        for j in np.where(valid)[0]:
            X = np.column_stack([np.ones(len(y))]
                                + ([cv] if cv is not None else [])
                                + [g[:, j]])
            val = ~np.isnan(X).any(1)
            b_, se = _logistic_irls(X[val], y[val])
            if se is None or se <= 0:
                valid[j] = False; continue
            z = b_ / se
            chi2[j] = z ** 2
            pv[j] = float(2 * stats.norm.sf(abs(z)))
        valid &= np.isfinite(pv)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return pd.DataFrame()
    return pd.DataFrame({
        "SNP": snp["SNP"].values[idx],
        "CHR": snp["CHR"].astype(str).values[idx],
        "BP": snp["BP"].astype(int).values[idx],
        "freq_A": pA[idx].round(4), "freq_B": pB[idx].round(4),
        "delta_freq": (pA - pB)[idx].round(4), "MAF": mj[idx].round(4),
        "chi2": chi2[idx].round(4), "P": pv[idx],
        "FST_snp": fv[idx].round(4),
        "n_A": nA[idx].astype(int), "n_B": nB[idx].astype(int),
    }).sort_values("P").reset_index(drop=True)

@cache_data
def gwas_pairwise(gt, snp_df, labels, ra, rb, maf_min=0.05,
                  method="allelic", covs=None):
    return _pairwise_core(gt, snp_df, labels, ra, rb, maf_min, method, covs)

@cache_data
def gwas_pairwise_group(gt, snp_df, groups, ga, gb, maf_min=0.05,
                        method="allelic", covs=None):
    return _pairwise_core(gt, snp_df, groups, ga, gb, maf_min, method, covs)

@cache_data
def get_pairs(labels):
    r = sorted(pd.unique(labels).astype(str))
    return r, [(r[i], r[j]) for i in range(len(r)) for j in range(i + 1, len(r))]

def _vc_delta(Y, X0, K):
    n = len(Y)
    d, U = np.linalg.eigh(K)
    d = np.maximum(d, 0)
    Yt = U.T @ Y
    X0t = U.T @ X0
    def nll(dl):
        if dl <= 0:
            return 1e18
        w = 1.0 / np.sqrt(dl * d + 1.0)
        Yw = Yt * w
        Xw = X0t * w[:, None]
        try:
            b, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
        except Exception:
            return 1e18
        s2 = float(np.sum((Yw - Xw @ b) ** 2) / n)
        return 1e18 if s2 <= 0 else 0.5 * (n * np.log(s2) + np.sum(np.log(dl * d + 1.0)))
    grid = np.logspace(-3, 3, 30)
    lls = np.array([nll(dl) for dl in grid])
    ib = int(np.argmin(lls))
    try:
        return max(float(minimize_scalar(
            nll, bounds=(grid[max(0, ib - 1)], grid[min(29, ib + 1)]),
            method="bounded").x), 1e-6)
    except Exception:
        return max(float(grid[ib]), 1e-6)

_GWAS_COLS = ["SNP_idx", "MAF", "BETA", "SE", "T", "P"]

@cache_data
def run_gwas(gt, pheno, K, covs, maf_min=0.05, miss_max=0.05,
             use_lmm=True, is_bin=False):
    pheno = np.asarray(pheno, dtype=float)
    valid = np.isfinite(pheno)
    if valid.sum() < 10:
        raise ValueError("<10 individus valides")
    Y = pheno[valid]
    gv = gt[valid]
    n = len(Y)
    miss = np.isnan(gv).mean(0)
    Gi = impute_mean(gv)
    p = Gi.mean(0) / 2.0
    mj = np.minimum(p, 1 - p)
    keep = np.where((miss <= miss_max) & (mj >= maf_min) & (Gi.std(0) > 1e-8))[0]
    if len(keep) == 0:
        return pd.DataFrame(columns=_GWAS_COLS)
    X0 = np.ones((n, 1))
    if covs is not None and np.size(covs) > 0:
        C = np.asarray(covs, dtype=float)[valid]
        C = C[:, np.nanstd(C, 0) > 1e-10]
        X0 = np.column_stack([X0, C])
    if is_bin:
        rows = []
        for j in keep:
            b_, se = _logistic_irls(
                np.column_stack([X0, Gi[:, j].astype(float)]), Y)
            if se is None or se <= 0:
                continue
            tv = b_ / se
            rows.append({"SNP_idx": int(j), "MAF": float(mj[j]),
                         "BETA": b_, "SE": se, "T": float(tv),
                         "P": float(2 * stats.norm.sf(abs(tv)))})
        return pd.DataFrame(rows, columns=_GWAS_COLS)
    lmm = use_lmm and isinstance(K, np.ndarray) and K.shape[0] == gt.shape[0]
    if lmm:
        Kv = K[np.ix_(valid, valid)]
        dl = _vc_delta(Y, X0, Kv)
        d, U = np.linalg.eigh(Kv)
        d = np.maximum(d, 0)
        w = 1.0 / np.sqrt(dl * d + 1.0)
        Yt = (U.T @ Y) * w
        X0t = (U.T @ X0) * w[:, None]
    else:
        Yt, X0t, U, w = Y, X0, None, None
    dof = n - X0t.shape[1] - 1
    if dof < 5:
        raise ValueError("Trop peu d'individus")
    Q, _ = np.linalg.qr(X0t)
    Yr = Yt - Q @ (Q.T @ Yt)
    yy = float(Yr @ Yr)
    out = []
    for s in range(0, len(keep), 4000):
        idx = keep[s:s + 4000]
        Gc = Gi[:, idx].astype(np.float64)
        Gt = (U.T @ Gc) * w[:, None] if lmm else Gc
        Gr = Gt - Q @ (Q.T @ Gt)
        gg = (Gr ** 2).sum(0)
        ok = gg > 1e-10
        gg_s = np.where(ok, gg, 1.0)
        beta = (Gr.T @ Yr) / gg_s
        s2 = np.clip((yy - beta ** 2 * gg_s) / dof, 1e-300, None)
        se = np.sqrt(s2 / gg_s)
        tv = beta / se
        pv = 2 * t_dist.sf(np.abs(tv), df=dof)
        out.append(pd.DataFrame({"SNP_idx": idx[ok], "MAF": mj[idx][ok],
                                 "BETA": beta[ok], "SE": se[ok],
                                 "T": tv[ok], "P": pv[ok]}))
    return pd.concat(out, ignore_index=True)

def annotate_gwas(r, snp_df):
    if r.empty:
        return r
    idx = r["SNP_idx"].astype(int).values
    r = r.copy()
    r["SNP"] = snp_df["SNP"].astype(str).values[idx]
    r["CHR"] = snp_df["CHR"].astype(str).values[idx]
    r["BP"] = snp_df["BP"].astype(float).astype(np.int64).values[idx]
    r["CM"] = (snp_df["CM"].astype(float).values[idx]
               if "CM" in snp_df.columns else 0.0)
    return r[["SNP_idx", "SNP", "CHR", "BP", "CM", "MAF", "BETA", "SE", "T", "P"]]

def lambda_gc(pvals):
    p = np.asarray(pvals)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10:
        return np.nan
    return float(np.median(stats.chi2.isf(p, df=1)) / stats.chi2.ppf(0.5, df=1))

def _bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(q, 0, 1)
    return out

def postprocess_gwas(r, apply_gc=True):
    if r is None or r.empty:
        return r
    r = r.copy()
    r["FDR"] = _bh(r["P"].values)
    r["log10P"] = -np.log10(r["P"].clip(1e-300, 1))
    if apply_gc:
        lam = lambda_gc(r["P"].values)
        if np.isfinite(lam) and lam > 1.0:
            r["P_GC"] = stats.chi2.sf(
                stats.chi2.isf(r["P"].clip(1e-300, 1), df=1) / lam, df=1)
            r["log10P_GC"] = -np.log10(r["P_GC"].clip(1e-300, 1))
    return r.sort_values("P").reset_index(drop=True)

def fine_mapping(r, window_mb=1.0, w_prior=0.04):
    if r is None or r.empty:
        return pd.DataFrame()
    d = r[np.isfinite(r["BETA"]) & np.isfinite(r["SE"]) & (r["SE"] > 0)].copy()
    if d.empty:
        return d
    lead = d.sort_values("P").iloc[0]
    reg = d[(d["CHR"].astype(str) == str(lead["CHR"]))
            & ((d["BP"] - lead["BP"]).abs() <= window_mb * 1e6)].copy()
    V = reg["SE"] ** 2
    z2 = reg["BETA"] ** 2 / V
    reg["logABF"] = 0.5 * np.log(V / (V + w_prior)) + 0.5 * z2 * w_prior / (V + w_prior)
    reg["ABF"] = np.exp(reg["logABF"] - reg["logABF"].max())
    reg["PIP"] = reg["ABF"] / reg["ABF"].sum()
    reg = reg.sort_values("PIP", ascending=False).reset_index(drop=True)
    reg["CUM_PIP"] = reg["PIP"].cumsum()
    return reg

def run_gblup(gt, pheno, kf=5, ids=None, seed=42):
    pheno = np.asarray(pheno, dtype=float)
    valid = np.isfinite(pheno)
    if valid.sum() < 20:
        raise ValueError("<20 phénotypes")
    Y = pheno[valid]
    n = len(Y)
    kf = max(2, min(kf, n - 1))
    X = impute_mean(gt[valid])
    p = X.mean(0) / 2.0
    Z = X - 2 * p
    G = (Z @ Z.T) / (2 * np.sum(p * (1 - p))) + 1e-5 * np.eye(n)
    d, U = np.linalg.eigh(G)
    d = np.maximum(d, 0)
    mu = Y.mean()
    Yt = U.T @ (Y - mu)
    def ll(h2):
        lam = (1 - h2) / h2
        return -0.5 * (np.sum(np.log(d + lam))
                       + n * np.log(np.sum(Yt ** 2 / (d + lam))))
    grid = np.linspace(0.01, 0.99, 50)
    ib = int(np.argmax([ll(h) for h in grid]))
    try:
        h2 = float(minimize_scalar(
            lambda h: -ll(h),
            bounds=(grid[max(0, ib - 1)], grid[min(49, ib + 1)]),
            method="bounded").x)
    except Exception:
        h2 = float(grid[ib])
    lam = (1 - h2) / h2
    gebv = U @ (d / (d + lam) * Yt)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, kf)
    yp = np.zeros(n)
    for te in folds:
        tr = np.setdiff1d(idx, te)
        mtr = Y[tr].mean()
        yp[te] = mtr + G[np.ix_(te, tr)] @ np.linalg.solve(
            G[np.ix_(tr, tr)] + lam * np.eye(len(tr)), Y[tr] - mtr)
    corr = float(np.corrcoef(Y, yp)[0, 1]) if n > 2 else 0.0
    names = np.asarray(ids)[valid] if ids is not None else np.where(valid)[0]
    return (pd.DataFrame({"IID": names, "Phenotype": Y, "GEBV": gebv}),
            {"h2": h2, "corr": corr, "r2": corr ** 2,
             "rmse": float(np.sqrt(np.mean((Y - yp) ** 2)))})

# ══════════════════════════════════════════════════════════════════════════════
# Combinaisons p-values + permutations
# ══════════════════════════════════════════════════════════════════════════════
def _merge_res(res_race, res_group):
    return (res_race[["SNP", "CHR", "BP", "P"]].rename(columns={"P": "P_race"})
            .merge(res_group[["SNP", "P"]].rename(columns={"P": "P_group"}),
                   on="SNP", how="inner"))

def combine_pvalues_fisher(res_race, res_group):
    if res_race is None or res_group is None:
        return None
    m = _merge_res(res_race, res_group)
    if m.empty:
        return None
    m["P_combined"] = [stats.combine_pvalues(r, method="fisher")[1]
                       for r in m[["P_race", "P_group"]].values]
    m["logP_combined"] = -np.log10(m["P_combined"].clip(1e-300, 1))
    return m.sort_values("P_combined").reset_index(drop=True)

def combine_stouffer(res_race, res_group, n_race, n_group):
    if res_race is None or res_group is None:
        return None
    m = _merge_res(res_race, res_group)
    if m.empty:
        return None
    wr, wg = np.sqrt(n_race), np.sqrt(n_group)
    zr = stats.norm.isf(m["P_race"].clip(1e-300, 1).values / 2)
    zg = stats.norm.isf(m["P_group"].clip(1e-300, 1).values / 2)
    m["P_combined"] = 2 * stats.norm.sf(
        np.abs((wr * zr + wg * zg) / np.sqrt(wr ** 2 + wg ** 2)))
    m["logP_combined"] = -np.log10(m["P_combined"].clip(1e-300, 1))
    return m.sort_values("P_combined").reset_index(drop=True)

@cache_data
def permutation_test_race(gt, snp_df, labels, ra, rb, n_perm=100, seed=42,
                          max_snp=5000, maf_min=0.05):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(str)
    if len(snp_df) > max_snp:
        idx = np.sort(rng.choice(len(snp_df), max_snp, replace=False))
        snp_df = snp_df.iloc[idx].reset_index(drop=True)
        gt = gt[:, idx]
    original = _pairwise_core(gt, snp_df, labels, ra, rb, maf_min)
    if original.empty:
        return pd.DataFrame(), 1.0, 0, [], 1.0
    n_obs = int((original["P"] < 0.05 / len(original)).sum())
    min_obs = original["P"].min()
    mask = np.isin(labels, [ra, rb])
    lab_sub = labels[mask].copy()
    gt_sub = gt[mask]
    counts, minps = [], []
    for _ in range(n_perm):
        rng.shuffle(lab_sub)
        r = _pairwise_core(gt_sub, snp_df, lab_sub, ra, rb, maf_min)
        counts.append(int((r["P"] < 0.05 / len(r)).sum()) if not r.empty else 0)
        minps.append(float(r["P"].min()) if not r.empty else 1.0)
    return (original,
            float((np.sum(np.array(counts) >= n_obs) + 1) / (n_perm + 1)),
            n_obs, counts,
            float((np.sum(np.array(minps) <= min_obs) + 1) / (n_perm + 1)))


# ══════════════════════════════════════════════════════════════════════════════
# VEP / conséquences
# ══════════════════════════════════════════════════════════════════════════════
_SEVERITY = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}

def traduire_consequence_vep(c):
    c = str(c).lower()
    if any(k in c for k in ("stop_gained", "frameshift", "start_lost", "stop_lost")):
        return "⛔ Exon (Non-sens / perte de codon)"
    if "missense" in c:
        return "🔴 Exon (Faux-sens - Protéine modifiée)"
    if "synonymous" in c:
        return "🟡 Exon (Silencieuse - Protéine inchangée)"
    if "splice" in c:
        return "⚠️ Jonction Exon/Intron"
    if "intron" in c:
        return "🟢 Intron (Régulation)"
    if "3_prime_utr" in c:
        return "🔵 Région 3'UTR"
    if "5_prime_utr" in c:
        return "🔵 Région 5'UTR"
    if "upstream" in c:
        return "🟣 Régulateur (Promoteur amont)"
    if "downstream" in c:
        return "🟣 Régulateur (Aval)"
    if "regulatory" in c or "tf_binding" in c:
        return "🟣 Région régulatrice"
    return "⚪ Intergénique"

def run_vep_api(items, species, batch=200):
    if not items:
        return pd.DataFrame()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    results = []
    def _post(url, payload, mapping):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2)))
                r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
            if r.status_code != 200:
                st.warning(f"⚠️ VEP HTTP {r.status_code} : {r.text[:150]}")
                return
            for item in r.json():
                sid = mapping.get(item.get("input"), item.get("input", "N/A"))
                if item.get("transcript_consequences"):
                    cons = sorted(item["transcript_consequences"],
                                  key=lambda x: _SEVERITY.get(x.get("impact", "MODIFIER"), 0),
                                  reverse=True)[0]
                    results.append({
                        "SNP": sid,
                        "GENE": cons.get("gene_symbol", cons.get("gene_id", "N/A")),
                        "CONSEQUENCE": ", ".join(cons.get("consequence_terms", ["N/A"])),
                        "IMPACT": cons.get("impact", "N/A"),
                        "AMINO_ACID_CHANGE": cons.get("amino_acids", "N/A"),
                        "SIFT": cons.get("sift_prediction", "N/A"),
                        "POLYPHEN": cons.get("polyphen_prediction", "N/A"),
                    })
                elif item.get("regulatory_feature_consequences"):
                    cons = item["regulatory_feature_consequences"][0]
                    results.append({"SNP": sid, "GENE": "Région régulatrice",
                                    "CONSEQUENCE": cons.get("consequence_terms", ["N/A"])[0],
                                    "IMPACT": "MODIFIER",
                                    "AMINO_ACID_CHANGE": "N/A",
                                    "SIFT": "N/A", "POLYPHEN": "N/A"})
                else:
                    results.append({"SNP": sid,
                                    "GENE": "Intergenic/Non-codant",
                                    "CONSEQUENCE": item.get("most_severe_consequence", "N/A"),
                                    "IMPACT": "MODIFIER",
                                    "AMINO_ACID_CHANGE": "N/A",
                                    "SIFT": "N/A", "POLYPHEN": "N/A"})
        except Exception as e:
            st.warning(f"⚠️ VEP réseau : {e}")
    reg = [(s, v) for s, k, v in items if k == "region"]
    ids = [(s, v) for s, k, v in items if k == "id"]
    for i in range(0, len(reg), batch):
        ch = reg[i:i + batch]
        _post(f"https://rest.ensembl.org/vep/{species}/region?regulatory=1",
              {"variants": [v for _, v in ch]},
              {v: s for s, v in ch})
        time.sleep(0.4)
    for i in range(0, len(ids), batch):
        ch = ids[i:i + batch]
        _post(f"https://rest.ensembl.org/vep/{species}/id?regulatory=1",
              {"ids": [v for _, v in ch]},
              {v: s for s, v in ch})
        time.sleep(0.4)
    return pd.DataFrame(results)

# ══════════════════════════════════════════════════════════════════════════════
# NCBI / UniProt / AlphaFold
# ══════════════════════════════════════════════════════════════════════════════
def fetch_ncbi_gene_info(gene_list, taxon):
    results = []
    for gene in gene_list:
        clean_gene = str(gene).strip()
        if not clean_gene or clean_gene.lower() in (
            "nan", "n/a", "unknown", "intergenic/non-codant", "région régulatrice"
        ):
            continue
        url = f"https://api.ncbi.nlm.nih.gov/datasets/v2/gene/symbol/{quote(clean_gene)}/taxon/{taxon}"
        try:
            r = requests.get(url, timeout=30, headers={"Accept": "application/json"})
            reports = (r.json().get("reports") or []) if r.status_code == 200 else []
            gi = (reports[0].get("gene") or {}) if reports else {}
            if not gi:
                results.append({"GENE": clean_gene, "Nom complet": "Non trouvé",
                                "Chromosome": "N/A", "Résumé": "N/A"})
                continue
            summ = gi.get("summary")
            if isinstance(summ, list):
                summ = " ".join(x.get("description", "")
                                for x in summ if isinstance(x, dict))
            results.append({
                "GENE": clean_gene,
                "Nom complet": gi.get("description", "N/A"),
                "Chromosome": (gi.get("chromosomes") or ["N/A"])[0],
                "Résumé": (str(summ)[:200] + "...") if summ else "N/A",
            })
        except Exception as e:
            results.append({"GENE": clean_gene, "Nom complet": "Erreur API",
                            "Chromosome": "N/A", "Résumé": str(e)})
        time.sleep(0.4)
    return pd.DataFrame(results)

# [FIX] normalisation accession + label espèce
def fetch_uniprot_id(gene_name, taxon):
    gene = str(gene_name).strip()
    if not gene or gene.lower() in ("nan", "n/a", "unknown"):
        return None, None
    for organism, label in ((taxon, "Espèce cible"), ("9606", "Humain (Orthologue)")):
        q = f"gene:{gene} AND organism_id:{organism}"
        url = f"https://rest.uniprot.org/uniprotkb/search?query={quote(q)}&format=json&size=1"
        try:
            r = requests.get(url, timeout=30, headers={"Accept": "application/json"})
            if r.status_code == 200 and r.json().get("results"):
                acc = r.json()["results"][0].get("primaryAccession")
                if acc:
                    return acc.split("-")[0].strip(), label
        except Exception:
            pass
    return None, None

# [FIX] debug + normalisation + distinction 404 vs pas de pdbUrl
def fetch_alphafold_pdb(uniprot_id, debug=False):
    if not uniprot_id:
        return None
    acc = str(uniprot_id).split("-")[0].strip()
    try:
        r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=30)
        if debug:
            st.caption(f"AlphaFold HTTP {r.status_code} pour `{acc}`")
        if r.status_code != 200:
            return None
        recs = r.json() or []
        if not recs:
            return None
        rec = next((x for x in recs if x.get("uniprotAccession") == acc), recs[0])
        url = rec.get("pdbUrl") or rec.get("cifUrl")
        if not url:
            if debug:
                st.caption("Réponse AlphaFold sans `pdbUrl` ni `cifUrl`.")
            return None
        pr = requests.get(url, timeout=60)
        if pr.status_code == 200:
            return pr.text
    except Exception as e:
        if debug:
            st.caption(f"AlphaFold échec : {type(e).__name__}: {e}")
    return None

def render_3d_structure(pdb_data, style="cartoon", color_scheme="spectrum",
                        highlight_res=None):
    sty = {style: ({"color": "spectrum"} if color_scheme == "spectrum"
                   else {"colorscheme": color_scheme})}
    hl = ""
    if highlight_res and str(highlight_res).strip().isdigit():
        hl = (f"viewer.addStyle({{resi: {int(highlight_res)}}}, "
              f"{{stick: {{colorscheme: 'redCarbon'}}}});")
    html = f"""<div id="viewport" style="width:100%;height:500px;position:relative;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      let viewer = $3Dmol.createViewer(document.getElementById("viewport"), {{backgroundColor: 'white'}});
      viewer.addModel({json.dumps(pdb_data)}, "pdb");
      viewer.setStyle({{}}, {json.dumps(sty)}); {hl}
      viewer.zoomTo(); viewer.render();
    </script>"""
    components.html(html, height=520)

# ══════════════════════════════════════════════════════════════════════════════
# Ensembl sequence / alignement / orthologues
# ══════════════════════════════════════════════════════════════════════════════
def fetch_ensembl_sequence(chrom, start, end, species):
    try:
        url = (f"https://rest.ensembl.org/sequence/region/{species}/"
               f"{chrom}:{start}..{end}:1?content-type=text/plain")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            return r.text.strip().upper()
    except Exception:
        pass
    return None

def align_and_highlight(seq_ref, seq_a, seq_b, name_a="Allèle A", name_b="Allèle B"):
    if not HAS_BIO:
        return "<p>⚠️ Biopython non installé.</p>"
    al = Align.PairwiseAligner()
    al.mode = "global"
    al.match_score = 2
    al.mismatch_score = -3
    al.open_gap_score = -5
    al.extend_gap_score = -2
    def fmt(seq, name, color):
        aln = al.align(seq_ref, seq)[0]
        ref_aln, seq_aln = str(aln[0]), str(aln[1])
        s = f"<b>{name:10}</b> | "
        for r, q in zip(ref_aln, seq_aln):
            if r == "-" or q == "-":
                s += f"<span style='color:orange;font-weight:bold;'>{q}</span>"
            elif r != q:
                s += f"<span style='color:red;font-weight:bold;'>{q}</span>"
            else:
                s += f"<span style='color:{color};'>{q}</span>"
        return s + "<br>", ref_aln
    ha, ref_aln = fmt(seq_a, name_a, "green")
    hb, _ = fmt(seq_b, name_b, "blue")
    ref_line = "".join(f"<span style='color:grey;'>{r}</span>" for r in ref_aln)
    return ("<div style='font-family:monospace;font-size:16px;line-height:1.5;'>"
            + ha + f"<b>{'Référence':10}</b> | " + ref_line + "<br>" + hb + "</div>")

def ortholog_lookup(chrom, bp, species, target="homo_sapiens"):
    base = "https://rest.ensembl.org"
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        genes = []
        for w in (0, 50000):
            r = requests.get(
                f"{base}/overlap/region/{species}/{chrom}:{max(1, bp - w)}-{bp + w}?feature=gene",
                headers=h, timeout=30)
            genes = r.json() if r.status_code == 200 else []
            if genes:
                break
        if not genes:
            return None
        g = next((x for x in genes if x.get("biotype") == "protein_coding"), genes[0])
        gid = g["id"]
        r = requests.get(
            f"{base}/homology/id/{species}/{gid}?target_species={target};type=orthologues",
            headers=h, timeout=60)
        data = r.json().get("data", []) if r.status_code == 200 else []
        homs = data[0].get("homologies", []) if data else []
        if not homs:
            return {"gene_src": g.get("external_name", gid), "gene_id": gid,
                    "human_gene": None}
        t = homs[0]["target"]
        r = requests.get(f"{base}/lookup/id/{t['id']}", headers=h, timeout=30)
        info = r.json() if r.status_code == 200 else {}
        return {"gene_src": g.get("external_name", gid), "gene_id": gid,
                "human_gene": info.get("display_name", t["id"]),
                "human_id": t["id"],
                "chr": info.get("seq_region_name"),
                "start": info.get("start"), "end": info.get("end"),
                "perc_id": t.get("perc_id"),
                "assembly": info.get("assembly_name")}
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# Annotation gènes (bp / cM)
# ══════════════════════════════════════════════════════════════════════════════
def annotate_genes(snp_df, gff, wk=50):
    if gff is None or gff.empty:
        return pd.DataFrame()
    w = wk * 1000
    res = []
    gchr = gff["CHR"].astype(str).values
    for _, s in snp_df.iterrows():
        pos = int(s.get("BP", s.get("START", 0)))
        ch = str(s["CHR"])
        ng = gff[(gchr == ch) & (gff["START"] <= pos + w) & (gff["END"] >= pos - w)]
        for _, g in ng.iterrows():
            d = (0 if g["START"] <= pos <= g["END"]
                 else min(abs(g["START"] - pos), abs(g["END"] - pos)))
            res.append({"SNP": s.get("SNP", f"{ch}:{pos}"), "CHR": ch,
                        "SNP_POS": pos, "GENE": g["GENE"],
                        "GENE_START": g["START"], "GENE_END": g["END"],
                        "DIST_KB": round(d / 1000.0, 2)})
    return pd.DataFrame(res)

def _snp_cm(snp_df_full, snp_id):
    try:
        row = snp_df_full[snp_df_full["SNP"].astype(str) == str(snp_id)]
        if row.empty:
            return None
        cm = row["CM"].iloc[0]
        return float(cm) if pd.notna(cm) else None
    except Exception:
        return None

def _cm_window_to_bp(snp_df_full, chrom, center_cm, window_cm):
    sub = snp_df_full[snp_df_full["CHR"].astype(str) == str(chrom)].copy()
    if sub.empty or np.all(sub["CM"].values.astype(float) == 0):
        return None
    sub = sub.sort_values("CM").reset_index(drop=True)
    cm_s = sub["CM"].values.astype(float)
    bp_s = sub["BP"].values.astype(float)
    lo = max(cm_s.min(), center_cm - window_cm)
    hi = min(cm_s.max(), center_cm + window_cm)
    a = float(np.interp(lo, cm_s, bp_s))
    b = float(np.interp(hi, cm_s, bp_s))
    return int(min(a, b)), int(max(a, b))

def _bp_to_cm(snp_df_full, chrom, bp):
    sub = snp_df_full[snp_df_full["CHR"].astype(str) == str(chrom)].copy()
    if sub.empty or np.all(sub["CM"].values.astype(float) == 0):
        return None
    sub = sub.sort_values("BP").reset_index(drop=True)
    return float(np.interp(bp, sub["BP"].values.astype(float),
                           sub["CM"].values.astype(float)))

def annotate_genes_cm(snp_target, snp_df_full, gff, window_cm=10,
                     window_kb_fallback=100):
    if gff is None or gff.empty:
        return pd.DataFrame()
    results = []
    has_cm = not np.all(snp_df_full["CM"].astype(float).values == 0)
    gchr = gff["CHR"].astype(str).values
    for _, s in snp_target.iterrows():
        chrom = str(s["CHR"])
        snp_id = s.get("SNP", "")
        bp_pos = int(s.get("BP", 0))
        cm_pos = s.get("CM", None)
        if cm_pos is None or pd.isna(cm_pos):
            cm_pos = _snp_cm(snp_df_full, snp_id)
        if (cm_pos is None or pd.isna(cm_pos)) and has_cm:
            cm_pos = _bp_to_cm(snp_df_full, chrom, bp_pos)
        bp_range = None
        method = ""
        if has_cm and cm_pos is not None and not pd.isna(cm_pos):
            bp_range = _cm_window_to_bp(snp_df_full, chrom, float(cm_pos), window_cm)
            if bp_range is not None:
                method = f"{window_cm} cM"
        if bp_range is None:
            w = window_kb_fallback * 1000
            bp_range = (bp_pos - w, bp_pos + w)
            method = f"±{window_kb_fallback} kb"
        ng = gff[(gchr == chrom) & (gff["END"] >= bp_range[0])
                 & (gff["START"] <= bp_range[1])]
        for _, g in ng.iterrows():
            dist_bp = (0 if g["START"] <= bp_pos <= g["END"]
                       else min(abs(g["START"] - bp_pos), abs(g["END"] - bp_pos)))
            dist_cm = None
            if method.endswith("cM"):
                cs = _bp_to_cm(snp_df_full, chrom, bp_pos)
                cg = _bp_to_cm(snp_df_full, chrom, (g["START"] + g["END"]) / 2)
                if cs is not None and cg is not None:
                    dist_cm = abs(cg - cs)
            results.append({
                "SNP": snp_id, "CHR": chrom,
                "SNP_POS_BP": bp_pos,
                "SNP_POS_CM": round(float(cm_pos), 2) if cm_pos is not None and not pd.isna(cm_pos) else None,
                "GENE": g["GENE"],
                "GENE_START": int(g["START"]), "GENE_END": int(g["END"]),
                "DIST_KB": round(dist_bp / 1000.0, 2),
                "DIST_CM": round(dist_cm, 2) if dist_cm is not None else None,
                "WINDOW": method,
            })
    df = pd.DataFrame(results)
    if not df.empty:
        sort_key = "DIST_CM" if has_cm else "DIST_KB"
        df = df.sort_values(["SNP", sort_key]).reset_index(drop=True)
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Enrichissement / STRING
# ══════════════════════════════════════════════════════════════════════════════
def run_enrichment(gene_list, gene_sets=None, organism="Human", outdir=None):
    try:
        import gseapy as gp
        genes = list({str(g).strip() for g in gene_list
                      if str(g).strip() and str(g) != "nan"})
        if len(genes) < 3:
            return pd.DataFrame()
        if gene_sets is None:
            gene_sets = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
        enr = gp.enrichr(gene_list=genes, gene_sets=gene_sets,
                         organism=organism, outdir=outdir, cutoff=1.0)
        if enr is None or enr.results is None:
            return pd.DataFrame()
        df = enr.results.copy()
        for old, new in {"Term": "TERM", "Adjusted P-value": "FDR",
                         "P-value": "PVAL", "Overlap": "OVERLAP",
                         "Genes": "GENES", "Combined Score": "SCORE"}.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        return (df.sort_values("FDR").reset_index(drop=True)
                if "FDR" in df.columns else df)
    except Exception as e:
        st.error(f"❌ Enrichr : {e}")
        return pd.DataFrame()

def query_string_network(genes, species, required_score=400):
    if not genes or len(genes) < 2:
        return pd.DataFrame(), pd.DataFrame()
    ids = "%0d".join([quote(str(g).strip()) for g in genes if str(g).strip()])
    url = (f"https://string-db.org/api/json/network?identifiers={ids}"
           f"&species={species}&required_score={required_score}"
           f"&caller_identity=livestock_genomics_pipeline")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame(), pd.DataFrame()
        edges = pd.DataFrame(data)[["preferredName_A", "preferredName_B", "score"]].drop_duplicates()
        edges.columns = ["SOURCE", "TARGET", "SCORE"]
        nodes = pd.DataFrame({"ID": sorted(set(edges["SOURCE"]).union(set(edges["TARGET"])))})
        return edges, nodes
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# Export PLINK / VCF
# ══════════════════════════════════════════════════════════════════════════════
def _alleles_for_export(snp_df, m):
    a_minor = (snp_df["A2"].astype(str).values[:m]
               if "A2" in snp_df.columns else np.array(["G"] * m))
    a_major = (snp_df["A1"].astype(str).values[:m]
               if "A1" in snp_df.columns else np.array(["A"] * m))
    return a_minor, a_major

def build_plink(gt, ind_df, snp_df, prefix="genome"):
    n, m = gt.shape
    nb = (n + 3) // 4
    code = np.full(gt.shape, 0b01, dtype=np.uint8)
    code[gt == 0] = 0b11
    code[gt == 1] = 0b10
    code[gt == 2] = 0b00
    code = np.vstack([code, np.zeros((nb * 4 - n, m), dtype=np.uint8)]).reshape(nb, 4, m)
    packed = code[:, 0, :] | (code[:, 1, :] << 2) | (code[:, 2, :] << 4) | (code[:, 3, :] << 6)
    a1, a2 = _alleles_for_export(snp_df, m)
    clean = lambda a: np.where(np.isin(a, ["", ".", "nan", "None"]), "0", a)
    bim = pd.DataFrame({
        "c": snp_df["CHR"].values[:m],
        "s": snp_df["SNP"].values[:m],
        "cm": snp_df["CM"].values[:m] if "CM" in snp_df.columns else 0.0,
        "bp": snp_df["BP"].astype(int).values[:m],
        "a1": clean(a1), "a2": clean(a2),
    })
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}.bed",
                    bytes([0x6C, 0x1B, 0x01]) + np.ascontiguousarray(packed.T).tobytes())
        zf.writestr(f"{prefix}.bim",
                    bim.to_csv(sep="\t", header=False, index=False))
        zf.writestr(f"{prefix}.fam",
                    "\n".join(f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9"
                              for _, r in ind_df.iterrows()) + "\n")
    return bio.getvalue()

def build_vcf(gt, ind_df, snp_df, proj="Genome"):
    m = gt.shape[1]
    alt, ref = _alleles_for_export(snp_df, m)
    acgt = np.array(["A", "C", "G", "T"])
    ref = np.where(np.isin(ref, acgt), ref, "A")
    alt = np.where(np.isin(alt, acgt), alt, "G")
    h = ["##fileformat=VCFv4.2", f"##source={proj}",
         '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
         "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
         + "\t".join(ind_df["IID"].astype(str))]
    lut = np.array(["0/0", "0/1", "1/1"])
    strs = np.where(np.isnan(gt), "./.",
                    lut[np.nan_to_num(gt, nan=0).astype(int).clip(0, 2)])
    b = []
    for j in range(m):
        r = snp_df.iloc[j]
        b.append(f"{r['CHR']}\t{int(r['BP'])}\t{r['SNP']}\t{ref[j]}\t{alt[j]}"
                 f"\t.\tPASS\t.\tGT\t" + "\t".join(strs[:, j]))
    return "\n".join(h + b) + "\n"

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTERS
# ══════════════════════════════════════════════════════════════════════════════
def plot_missing(mi, ms):
    f = make_subplots(rows=1, cols=2, subplot_titles=("Missing/Ind", "Missing/SNP"))
    f.add_trace(go.Histogram(x=mi, nbinsx=60, marker_color="skyblue"), 1, 1)
    f.add_trace(go.Histogram(x=ms, nbinsx=60, marker_color="coral"), 1, 2)
    f.update_layout(height=400, showlegend=False)
    return f

def plot_pca(s, v, l):
    fig = px.scatter(pd.DataFrame({"PC1": s[:, 0], "PC2": s[:, 1], "Pop": l}),
                     x="PC1", y="PC2", color="Pop",
                     title=f"PCA PC1({v[0]:.1f}%) vs PC2({v[1]:.1f}%)",
                     height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    return fig

def plot_mds(m, l):
    fig = px.scatter(pd.DataFrame({"MDS1": m[:, 0], "MDS2": m[:, 1], "Pop": list(l)}),
                     x="MDS1", y="MDS2", color="Pop",
                     title="MDS (distance IBS)", height=520)
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
    return fig

def _cum_positions(df, poscol):
    df = (df.assign(_k=df["CHR"].map(_chk_num), _s=df["CHR"].astype(str))
            .sort_values(["_k", "_s", poscol], kind="stable")
            .reset_index(drop=True))
    offs = {}
    ticks = []
    cur = 0.0
    for c, sub in df.groupby("CHR", sort=False):
        mx = float(sub[poscol].max()) / 1e6
        offs[c] = cur
        ticks.append((str(c), cur + mx / 2))
        cur += mx + 5
    df["_x"] = df["CHR"].map(offs) + df[poscol].astype(float) / 1e6
    return df, ticks

def plot_manhattan(df, yc="P", title="", thr=None):
    df = df.copy()
    ylab = yc
    if yc in ("P", "P_GC"):
        df["_y"] = -np.log10(np.clip(df[yc], 1e-300, 1))
        yc = "_y"; ylab = "-log10(P)"
    df = df[np.isfinite(df[yc])]
    if df.empty:
        return go.Figure()
    df, ticks = _cum_positions(df, "BP")
    fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR", sort=False)):
        fig.add_trace(go.Scatter(
            x=sub["_x"], y=sub[yc], mode="markers",
            marker=dict(size=5, color="#2c3e50" if i % 2 == 0 else "#7f8c8d"),
            text=sub["SNP"], hoverinfo="text+y", showlegend=False))
    if thr:
        fig.add_hline(y=thr, line_dash="dash", line_color="red")
    fig.update_layout(
        title=title,
        xaxis=dict(title="Chromosome", tickmode="array",
                   tickvals=[t[1] for t in ticks], ticktext=[t[0] for t in ticks]),
        yaxis_title=ylab, height=500)
    return fig

def plot_qq(pvals, title="QQ-plot"):
    p = np.sort(np.asarray(pvals)[np.isfinite(pvals)])
    p = p[(p > 0) & (p <= 1)]
    n = len(p)
    if n == 0:
        return go.Figure()
    exp = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    obs = -np.log10(p)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=exp, y=obs, mode="markers",
                             marker=dict(size=4), name="obs"))
    fig.add_trace(go.Scatter(x=[0, exp.max()], y=[0, exp.max()], mode="lines",
                             line=dict(color="red", dash="dash"), name="attendu"))
    fig.update_layout(title=title, xaxis_title="-log10(P) attendu",
                      yaxis_title="-log10(P) observé", height=420)
    return fig

def plot_ld(ld):
    if ld is None or ld.empty or len(ld) < 2:
        return None
    df = ld.sort_values("dist_kb").copy()
    df = df[df["dist_kb"] > 0]
    if df.empty:
        return None
    fig = px.line(df, x="dist_kb", y="r2", markers=True,
                  title="Décroissance du LD (r² moyen)",
                  labels={"dist_kb": "Distance (kb)", "r2": "r²"}, height=450)
    fig.update_layout(xaxis_type="log")
    return fig

def plot_admix(Q, labs, pops, K):
    df = pd.DataFrame(Q, columns=[f"K{k + 1}" for k in range(K)])
    df["IID"] = labs; df["Pop"] = pops; df["_d"] = Q.argmax(1)
    df["_po"] = pd.Categorical(pops, categories=sorted(set(pops)))
    df = df.sort_values(["_po", "_d"]).reset_index(drop=True)
    df["x"] = np.arange(len(df))
    fig = go.Figure()
    for k in range(K):
        fig.add_trace(go.Bar(x=df["x"], y=df[f"K{k + 1}"], name=f"K{k + 1}",
                             marker_color=px.colors.qualitative.Set2[k % 8]))
    fig.update_layout(barmode="stack", title=f"Admixture (NMF) K={K}", height=500)
    return fig, df

def plot_gblup(df):
    fig = px.scatter(df, x="Phenotype", y="GEBV", title="GBLUP")
    fig.update_traces(marker=dict(size=10, color="royalblue"))
    if len(df) > 1:
        try:
            z = np.polyfit(df["Phenotype"], df["GEBV"], 1)
            xr = np.linspace(df["Phenotype"].min(), df["Phenotype"].max(), 100)
            fig.add_trace(go.Scatter(x=xr, y=np.poly1d(z)(xr), mode="lines",
                                     name="OLS", line=dict(color="red", dash="dash")))
        except Exception:
            pass
    return fig

def plot_pip(df):
    return px.bar(df.head(20), x="SNP", y="PIP",
                  title="PIP (probabilité d'inclusion a posteriori)",
                  color="PIP", color_continuous_scale="Viridis")

def plot_win_manh(df, vc="MEAN_FST", title="", sigma=3.0):
    df = df[np.isfinite(df[vc])].copy()
    if df.empty:
        return None
    df, ticks = _cum_positions(df, "MID")
    fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR", sort=False)):
        fig.add_trace(go.Scatter(
            x=sub["_x"], y=sub[vc], mode="markers",
            marker=dict(size=7, color="#2c3e50" if i % 2 == 0 else "#7f8c8d",
                        line=dict(width=0.5, color="white")),
            text=[f"{c}:{int(m / 1e6)}Mb<br>{vc}={v:.4f}"
                  for m, v in zip(sub["MID"], sub[vc])],
            hoverinfo="text", showlegend=False))
    mu, sd = df[vc].mean(), df[vc].std()
    fig.add_hline(y=mu + sigma * sd, line_dash="dash", line_color="red",
                  annotation_text=f"μ+{sigma}σ")
    fig.update_layout(
        title=title,
        xaxis=dict(title="Chromosome", tickmode="array",
                   tickvals=[t[1] for t in ticks], ticktext=[t[0] for t in ticks]),
        yaxis_title=vc, height=520)
    return fig

def plot_gene_track(si, gff, wk=50):
    if gff is None or gff.empty:
        return None
    ch = str(si["CHR"])
    pos = int(si.get("BP", si.get("START", 0)))
    w = wk * 1000
    rg = gff[(gff["CHR"].astype(str) == ch)
             & (gff["START"] <= pos + w) & (gff["END"] >= pos - w)]
    fig = go.Figure()
    for i, (_, g) in enumerate(rg.iterrows()):
        fig.add_trace(go.Scatter(
            x=[g["START"] / 1e6, g["END"] / 1e6], y=[i, i],
            mode="lines", line=dict(width=10, color="royalblue"),
            name=g["GENE"], text=[g["GENE"]] * 2, hoverinfo="text"))
    fig.add_vline(x=pos / 1e6, line_dash="dash", line_color="red",
                  annotation_text=f"SNP: {si.get('SNP', '')}")
    fig.update_layout(title=f"Gènes CHR{ch}", xaxis_title="Pos (Mb)",
                      yaxis=dict(showticklabels=False),
                      height=400 + len(rg) * 20)
    return fig

def plot_string_network(edges, nodes):
    if edges is None or edges.empty:
        return None
    try:
        import networkx as nx
    except ImportError:
        st.error("⚠️ `networkx` non installé.")
        return None
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(r["SOURCE"], r["TARGET"], weight=float(r["SCORE"]))
    pos = nx.spring_layout(G, seed=42, k=0.6)
    ex, ey = [], []
    for s, t in G.edges():
        x0, y0 = pos[s]; x1, y1 = pos[t]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                             line=dict(width=0.6, color="#cccccc"),
                             hoverinfo="skip", showlegend=False))
    deg = [d for _, d in G.degree()]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode="markers+text",
        marker=dict(size=[8 + 3 * d for d in deg], color=deg,
                    colorscale="Viridis", line=dict(width=1, color="white")),
        text=list(G.nodes()), textposition="top center", hoverinfo="text"))
    fig.update_layout(showlegend=False, height=600,
                      xaxis=dict(showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(showgrid=False, zeroline=False, visible=False))
    return fig

# [FIX] plot_tree complété (l'original était tronqué)
def plot_tree(Z, labels, support=None, title="", mid=None):
    n = len(labels)
    labels = [str(l) for l in labels]
    lo = leaves_list(Z)
    ly = {int(l): i for i, l in enumerate(lo)}
    cy = {i: ly[i] for i in range(n)}
    cx = {i: 0.0 for i in range(n)}
    fig = go.Figure()
    for k, row in enumerate(Z):
        i, j, d = int(row[0]), int(row[1]), float(row[2])
        yi, yj = cy[i], cy[j]
        xi, xj = cx[i], cx[j]
        if support is not None and k < len(support):
            s = support[k]
            col = "#2ecc71" if s >= 70 else "#f39c12" if s >= 50 else "#e74c3c"
        else:
            col = "#3498db"
        for xs, ys in (([xi, d], [yi, yi]), ([xj, d], [yj, yj]), ([d, d], [yi, yj])):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     line=dict(color=col, width=2),
                                     hoverinfo="skip", showlegend=False))
        cy[n + k] = (yi + yj) / 2
        cx[n + k] = d
        if support is not None and k < len(support):
            fig.add_annotation(x=d, y=(yi + yj) / 2,
                               text=f"<b>{int(support[k])}</b>",
                               showarrow=False,
                               font=dict(size=10, color=col),
                               bgcolor="white", borderpad=1)
    for i, lab in enumerate(labels):
        fig.add_annotation(x=0, y=ly[i], text=f"<b>{lab}</b>",
                           showarrow=False, xanchor="right",
                           font=dict(size=11))
    if mid is not None:
        fig.add_vline(x=mid, line_dash="dot", line_color="grey",
                      annotation_text=f"midpoint={mid:.4f}")
    fig.update_layout(
        title=title, height=max(420, 30 * n),
        xaxis_title="Distance (Reynolds)",
        yaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
        showlegend=False, margin=dict(l=140))
    return fig

# [FIX] plotters manquants
def plot_ne(ne_df):
    if ne_df is None or ne_df.empty:
        return None
    fig = px.line(ne_df, x="GenAgo", y="Ne", markers=True,
                  title="Effectif efficace historique (Ne)",
                  labels={"GenAgo": "Générations dans le passé", "Ne": "Ne"})
    fig.update_layout(xaxis_type="log", yaxis_type="log", height=460)
    return fig

def plot_roh(roh_df, f_roh, iids):
    if roh_df is None or roh_df.empty:
        return None
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Longueurs ROH", "F_ROH par individu"))
    fig.add_trace(go.Histogram(x=roh_df["length_kb"] / 1000,
                               nbinsx=40, marker_color="teal", name="ROH"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=[str(i) for i in iids], y=f_roh,
                         marker_color="coral", name="F_ROH"),
                  row=1, col=2)
    fig.update_xaxes(title_text="Longueur (Mb)", row=1, col=1)
    fig.update_xaxes(title_text="Individu", row=1, col=2, showticklabels=False)
    fig.update_yaxes(title_text="Nombre", row=1, col=1)
    fig.update_yaxes(title_text="F_ROH", row=1, col=2)
    fig.update_layout(height=440, showlegend=False)
    return fig

def plot_tajima(tj):
    if tj is None or tj.empty:
        return None
    df = tj[np.isfinite(tj["TAJIMA_D"])].copy()
    if df.empty:
        return None
    fig = px.scatter(df, x="MID", y="TAJIMA_D", color="CHR",
                     title="Tajima's D — scan génomique",
                     hover_data=["CHR", "START", "END", "N_SNP", "S"])
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="D = -2")
    fig.add_hline(y=+2, line_dash="dash", line_color="red", annotation_text="D = +2")
    fig.update_layout(height=480, showlegend=False)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# UI STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "gt": None, "ind": None, "snp": None, "pheno": None, "gff": None,
        "gmt": None, "species": None, "cfg": None, "qc_info": None,
        "pca": None, "pca_var": None, "diversity": None, "shannon": None,
        "fst_mat": None, "fst_pops": None, "win_fst": None,
        "gwas_res": None, "gblup_res": None, "tree": None,
        "ld": None, "ne": None, "roh": None, "froh": None, "tajima": None,
        "qc_params": None, "qc_auto": True, "qc_rem": True,
        "gwas_maf": 0.05, "gwas_lmm": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _reset_downstream():
    for k in ("pca", "pca_var", "diversity", "shannon", "fst_mat", "fst_pops",
              "win_fst", "gwas_res", "gblup_res", "tree", "ld", "ne", "roh",
              "froh", "tajima", "qc_info"):
        st.session_state[k] = None

def _sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        species = st.selectbox("Espèce", list(SPECIES_CONFIG.keys()))
        cfg = SPECIES_CONFIG[species]
        st.session_state.species = species
        st.session_state.cfg = cfg

        st.subheader("🧹 QC")
        qc_geno = st.slider("geno (missing/SNP)", 0.0, 1.0, 0.10, 0.01)
        qc_mind = st.slider("mind (missing/ind)", 0.0, 1.0, 0.20, 0.01)
        qc_maf = st.slider("MAF minimum", 0.0, 0.5, 0.05, 0.01)
        qc_hwe = st.number_input("HWE p-value", min_value=0.0, max_value=1.0,
                                 value=1e-6, step=1e-7, format="%.2e")
        if qc_hwe <= 0:
            qc_hwe = 1e-6
        qc_hwe_pop = st.checkbox("HWE par population", value=False)
        qc_het_sd = st.slider("Z het outlier", 1.0, 6.0, 3.0, 0.5)
        keep_auto = st.checkbox("Autosomes seulement", value=True)
        rem_outliers = st.checkbox("Retirer outliers het", value=True)

        st.session_state.qc_params = dict(
            geno=qc_geno, mind=qc_mind, maf=qc_maf, hwe=qc_hwe,
            hwe_pop=qc_hwe_pop, hwe_min_n=10, hwe_k=1, het_sd=qc_het_sd)
        st.session_state.qc_auto = keep_auto
        st.session_state.qc_rem = rem_outliers

        st.subheader("🔬 GWAS")
        st.session_state.gwas_maf = st.slider("MAF min GWAS", 0.0, 0.5, 0.05, 0.01)
        st.session_state.gwas_lmm = st.checkbox("Utiliser LMM", value=True)

def _require_data():
    if st.session_state.gt is None:
        st.warning("⚠️ Chargez d'abord des données dans l'onglet **Import**.")
        return False
    return True

def _tab_import():
    st.header("📁 Import des données")
    col1, col2 = st.columns(2)
    with col1:
        ped_file = st.file_uploader("Fichier PED / genotype",
                                    type=["ped", "txt", "gz"])
        map_file = st.file_uploader("Fichier MAP", type=["map", "txt", "gz"])
        vcf_file = st.file_uploader("Fichier VCF", type=["vcf", "gz"])
    with col2:
        pheno_file = st.file_uploader("Phénotypes (CSV/TSV)",
                                       type=["csv", "txt", "tsv", "gz"])
        gff_file = st.file_uploader("GFF3", type=["gff", "gff3", "gz"])
        gmt_file = st.file_uploader("GMT (gene sets)", type=["gmt", "txt"])

    if ped_file:
        with st.expander("🔍 Diagnostic PED", expanded=False):
            raw = ped_file.getvalue()
            st.json(diagnose_ped(raw, ped_file.name))
            for l in preview_lines(raw, 4):
                st.code(l)
    if map_file:
        with st.expander("🔍 Diagnostic MAP", expanded=False):
            raw = map_file.getvalue()
            st.json(diagnose_map(raw, map_file.name))
            for l in preview_lines(raw, 4):
                st.code(l)

    if st.button("🚀 Charger les données", type="primary"):
        try:
            snp_df = None
            gt = ind_df = None
            if vcf_file:
                with st.spinner("Lecture VCF..."):
                    raw = (gzip.decompress(vcf_file.getvalue())
                           if vcf_file.name.endswith(".gz")
                           else vcf_file.getvalue())
                    gt, ind_df, snp_df = parse_vcf(raw)
            elif ped_file:
                if map_file:
                    snp_df = parse_map(map_file.getvalue())
                with st.spinner("Lecture PED..."):
                    n_map = len(snp_df) if snp_df is not None else None
                    gt, ind_df, a1, a2 = parse_ped(ped_file.getvalue(), n_map)
                if snp_df is None:
                    snp_df = pd.DataFrame({
                        "CHR": ["1"] * gt.shape[1],
                        "SNP": [f"S{i}" for i in range(gt.shape[1])],
                        "CM": 0.0,
                        "BP": np.arange(gt.shape[1]) * 10000,
                        "A1": a1, "A2": a2,
                    })
                else:
                    snp_df = snp_df.iloc[:gt.shape[1]].reset_index(drop=True)
                    snp_df["A1"] = a1[:len(snp_df)]
                    snp_df["A2"] = a2[:len(snp_df)]
            else:
                st.warning("Chargez un VCF ou un PED/MAP.")
                return
            gt, ind_df, snp_df = sync_dims(gt, ind_df, snp_df)
            ind_df = rec_fid(ind_df)
            st.session_state.gt = gt
            st.session_state.ind = ind_df
            st.session_state.snp = snp_df
            _reset_downstream()

            if pheno_file:
                ph = load_pheno(pheno_file.getvalue(), pheno_file.name)
                matched, id_col = match_pheno(ind_df, ph)
                st.session_state.pheno = matched
                st.success(f"Phénotypes appariés via `{id_col}`")

            if gff_file:
                gff_raw = (gzip.decompress(gff_file.getvalue())
                           if gff_file.name.endswith(".gz")
                           else gff_file.getvalue())
                st.session_state.gff = load_gff(gff_raw)

            if gmt_file:
                st.session_state.gmt = parse_gmt(gmt_file.getvalue())

            st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} SNPs chargés.")
        except Exception as e:
            st.error(f"❌ {type(e).__name__}: {e}")

    if st.button("🎲 Générer données de démonstration"):
        cfg = st.session_state.cfg
        gt, ind_df, snp_df = gen_demo(150, 800, 4,
                                       autosomes=cfg["autosomes"],
                                       ars=cfg["ars"])
        st.session_state.gt = gt
        st.session_state.ind = ind_df
        st.session_state.snp = snp_df
        _reset_downstream()
        st.success("✅ Données démo chargées.")

def _tab_qc():
    st.header("🧹 Contrôle qualité")
    if not _require_data():
        return
    gt = st.session_state.gt
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Individus", gt.shape[0])
    c2.metric("SNPs", gt.shape[1])
    c3.metric("Chr", st.session_state.snp["CHR"].nunique())
    c4.metric("Missing %", f"{100 * np.isnan(gt).mean():.2f}%")

    st.plotly_chart(plot_missing(miss_ind(gt), miss_snp(gt)),
                    use_container_width=True)

    if st.button("🧹 Appliquer QC", type="primary"):
        try:
            gt, ind_df, snp_df, info = apply_qc(
                st.session_state.gt, st.session_state.ind, st.session_state.snp,
                st.session_state.qc_params,
                st.session_state.cfg["autosomes"],
                keep_autosomes=st.session_state.qc_auto,
                rem_out=st.session_state.qc_rem)
            st.session_state.gt = gt
            st.session_state.ind = ind_df
            st.session_state.snp = snp_df
            st.session_state.qc_info = info
            _reset_downstream()
            st.session_state.qc_info = info
            st.success(f"✅ Après QC : {gt.shape[0]} ind × {gt.shape[1]} SNPs.")
        except Exception as e:
            st.error(f"❌ {type(e).__name__}: {e}")

    if st.session_state.qc_info:
        st.json(st.session_state.qc_info)

def _tab_structure():
    st.header("🧬 Structure des populations")
    if not _require_data():
        return
    gt, ind_df, snp_df = st.session_state.gt, st.session_state.ind, st.session_state.snp
    labels = ind_df["FID"].astype(str).values

    method = st.radio("Méthode", ["PCA", "MDS", "Admixture (NMF)"], horizontal=True)

    if method == "PCA":
        if st.button("Calculer PCA"):
            s, v = pca_analysis(gt, nc=10)
            st.session_state.pca, st.session_state.pca_var = s, v
        if st.session_state.pca is not None:
            st.plotly_chart(
                plot_pca(st.session_state.pca, st.session_state.pca_var, labels),
                use_container_width=True)
            st.dataframe(
                pd.DataFrame({"PC": range(1, len(st.session_state.pca_var) + 1),
                              "Var%": st.session_state.pca_var.round(2)}),
                use_container_width=True)

    elif method == "MDS":
        if st.button("Calculer MDS"):
            m = mds_analysis(gt, nc=5)
            st.plotly_chart(plot_mds(m, labels), use_container_width=True)

    else:
        K = st.slider("K", 2, 8, 3)
        if st.button("Calculer Admixture"):
            Q, _ = admixture_nmf(gt, K=K)
            fig, df = plot_admix(Q, ind_df["IID"].values, labels, K)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.drop(columns=["_d", "_po", "x"]),
                         use_container_width=True)
        if st.button("Évaluer K optimal"):
            st.dataframe(choose_best_k(gt, 2, 8), use_container_width=True)

    st.subheader("🌳 Arbre (Reynolds + bootstrap)")
    if st.button("Construire arbre"):
        try:
            res = upgma_bootstrap(gt, snp_df, labels, n_boot=100)
            st.session_state.tree = res
            fig = plot_tree(res["Z"], res["pops"],
                            support=res["support"],
                            title="UPGMA (bootstrap)")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.code(res["newick"], language="text")
        except Exception as e:
            st.error(f"❌ {e}")

def _tab_diversity():
    st.header("📊 Diversité & différenciation")
    if not _require_data():
        return
    gt, ind_df = st.session_state.gt, st.session_state.ind
    labels = ind_df["FID"].astype(str).values

    st.subheader("Diversité intra-population")
    if st.button("Calculer diversité"):
        st.session_state.diversity = compute_diversity(gt, labels)
        st.session_state.shannon = shannon_weaver(gt, labels)
    if st.session_state.diversity is not None:
        st.dataframe(st.session_state.diversity, use_container_width=True)
    if st.session_state.shannon is not None:
        st.dataframe(st.session_state.shannon, use_container_width=True)

    st.subheader("Fst (Weir & Cockerham)")
    if st.button("Calculer Fst global + matrice"):
        M, pops = fst_pairwise(gt, labels)
        st.session_state.fst_mat = M
        st.session_state.fst_pops = pops
        fig = px.imshow(M, x=pops, y=pops, text_auto=".3f",
                        title="Fst matrice")
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state.fst_mat is not None:
        w = st.slider("Fenêtre SNPs", 10, 200, 50)
        if st.button("Fst fenêtré + outliers"):
            wf = windowed_fst(gt, st.session_state.snp, labels,
                              w=w, step=max(1, w // 2))
            st.session_state.win_fst = wf
            if not wf.empty:
                fig = plot_win_manh(detect_outliers(wf), vc="MEAN_FST",
                                    title="Fst fenêtré")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(detect_outliers(wf).head(30),
                             use_container_width=True)

def _tab_ld():
    st.header("🔗 LD, Ne et ROH")
    if not _require_data():
        return
    gt, snp_df = st.session_state.gt, st.session_state.snp

    st.subheader("Décroissance du LD")
    if st.button("Calculer LD decay"):
        ld = ld_decay(gt, snp_df)
        st.session_state.ld = ld
        fig = plot_ld(ld)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ld, use_container_width=True)

    st.subheader("Ne historique")
    if st.button("Estimer Ne"):
        ne = ne_historical(gt, snp_df)
        st.session_state.ne = ne
        fig = plot_ne(ne)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("ROH")
    min_snps = st.slider("min_snps ROH", 10, 100, 30)
    min_kb = st.slider("min_kb ROH", 100, 5000, 500)
    if st.button("Détecter ROH"):
        roh, froh = detect_roh(gt, snp_df["CHR"].values, snp_df["BP"].values,
                               min_snps=min_snps, min_kb=min_kb)
        st.session_state.roh = roh
        st.session_state.froh = froh
        if not roh.empty:
            fig = plot_roh(roh, froh, st.session_state.ind["IID"].values)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(roh.head(50), use_container_width=True)

    st.subheader("Tajima's D")
    if st.button("Scan Tajima"):
        w = 25
        tj = tajimas_scan(gt, snp_df, w=w, step=max(1, w // 2))
        st.session_state.tajima = tj
        fig = plot_tajima(tj)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tj.head(50), use_container_width=True)

def _tab_gwas():
    st.header("🎯 GWAS")
    if not _require_data():
        return
    gt, ind_df, snp_df = st.session_state.gt, st.session_state.ind, st.session_state.snp
    pheno = st.session_state.pheno
    if pheno is None:
        st.warning("Chargez des phénotypes dans l'onglet **Import**.")
        return

    numeric_cols = pheno.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("Aucune colonne phénotypique numérique.")
        return
    pheno_col = st.selectbox("Phénotype", numeric_cols)

    mode = st.radio("Mode",
                    ["GWAS par phénotype (LMM)", "GWAS par paires de races"],
                    horizontal=True)
    use_lmm = st.session_state.gwas_lmm
    maf_min = st.session_state.gwas_maf

    if mode == "GWAS par phénotype (LMM)":
        if st.button("Lancer GWAS", type="primary"):
            try:
                y = pheno[pheno_col].values
                K = grm_vanraden(gt) if use_lmm else None
                res = run_gwas(gt, y, K, covs=None,
                               maf_min=maf_min, use_lmm=use_lmm)
                if res.empty:
                    st.warning("Aucun SNP testé.")
                    return
                res = annotate_gwas(res, snp_df)
                res = postprocess_gwas(res)
                st.session_state.gwas_res = res
                st.success(f"✅ {len(res)} SNPs testés.")
                st.plotly_chart(plot_manhattan(res, yc="P",
                                title=f"Manhattan — {pheno_col}"),
                                use_container_width=True)
                st.plotly_chart(plot_qq(res["P"].values), use_container_width=True)
                st.dataframe(res.head(50), use_container_width=True)
            except Exception as e:
                st.error(f"❌ {type(e).__name__}: {e}")

        if st.session_state.gwas_res is not None:
            if st.button("Fine-mapping (ABF)"):
                fm = fine_mapping(st.session_state.gwas_res)
                if not fm.empty:
                    st.plotly_chart(plot_pip(fm), use_container_width=True)
                    st.dataframe(fm, use_container_width=True)
    else:
        labels = ind_df["FID"].astype(str).values
        _, pairs = get_pairs(labels)
        pair = st.selectbox("Paire", [f"{a} vs {b}" for a, b in pairs])
        ga, gb = [p.strip() for p in pair.split(" vs ")]
        if st.button("Lancer GWAS par paire"):
            try:
                res = gwas_pairwise(gt, snp_df, labels, ga, gb, maf_min=maf_min)
                if res.empty:
                    st.warning("Aucun SNP testé.")
                    return
                res = postprocess_gwas(res)
                st.session_state.gwas_res = res
                st.plotly_chart(plot_manhattan(res, yc="P",
                                title=f"{ga} vs {gb}"), use_container_width=True)
                st.dataframe(res.head(50), use_container_width=True)
            except Exception as e:
                st.error(f"❌ {e}")

def _tab_gblup():
    st.header("💾 GBLUP (valeurs génomiques)")
    if not _require_data():
        return
    pheno = st.session_state.pheno
    if pheno is None:
        st.warning("Chargez des phénotypes d'abord.")
        return
    numeric_cols = pheno.select_dtypes(include=[np.number]).columns.tolist()
    pheno_col = st.selectbox("Phénotype", numeric_cols)
    kf = st.slider("Folds (CV)", 2, 10, 5)
    if st.button("Calculer GBLUP", type="primary"):
        try:
            df, stats_ = run_gblup(st.session_state.gt,
                                   pheno[pheno_col].values,
                                   kf=kf, ids=pheno["IID"].values)
            st.session_state.gblup_res = df
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("h²", f"{stats_['h2']:.3f}")
            c2.metric("Corrélation", f"{stats_['corr']:.3f}")
            c3.metric("R²", f"{stats_['r2']:.3f}")
            c4.metric("RMSE", f"{stats_['rmse']:.3f}")
            st.plotly_chart(plot_gblup(df), use_container_width=True)
            st.dataframe(df.head(50), use_container_width=True)
        except Exception as e:
            st.error(f"❌ {e}")

def _top_with_alleles(top, snp_df):
    if "A1" in snp_df.columns and "A2" in snp_df.columns:
        alleles = snp_df[["SNP", "A1", "A2"]].drop_duplicates("SNP")
        return top.merge(alleles, on="SNP", how="left")
    return top

def _tab_annotation():
    st.header("🔬 Annotation fonctionnelle")
    if not _require_data():
        return
    snp_df = st.session_state.snp
    cfg = st.session_state.cfg
    gwas = st.session_state.gwas_res

    top = gwas.head(50) if (gwas is not None and not gwas.empty) else snp_df.head(50)
    top = _top_with_alleles(top, snp_df)

    # --- Gènes depuis GFF3
    gene_names = []
    if st.session_state.gff is not None:
        st.subheader("Annotation gènes (GFF3)")
        wk = st.slider("Fenêtre (kb)", 5, 500, 50)
        ann = annotate_genes(top, st.session_state.gff, wk=wk)
        if not ann.empty:
            gene_names = ann["GENE"].dropna().astype(str).unique().tolist()
            st.dataframe(ann, use_container_width=True)
            for _, r in ann.head(3).iterrows():
                fig = plot_gene_track(r, st.session_state.gff, wk=wk)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    # --- VEP Ensembl
    st.subheader("VEP (Ensembl)")
    if st.button("Annoter via VEP"):
        items = []
        for _, r in top.head(200).iterrows():
            a1 = str(r.get("A1", "A")) if "A1" in r else "A"
            a2 = str(r.get("A2", "G")) if "A2" in r else "G"
            variant = f"{r['CHR']} {int(r['BP'])} . {a1} {a2} . . ."
            items.append((r["SNP"], "region", variant))
        try:
            vep = run_vep_api(items, cfg["vep_species"])
            if not vep.empty:
                vep["TRADUCTION"] = vep["CONSEQUENCE"].apply(traduire_consequence_vep)
                st.dataframe(vep, use_container_width=True)
        except Exception as e:
            st.error(f"❌ {e}")

    # --- NCBI Gene Info
    st.subheader("NCBI Gene Info")
    if st.button("Interroger NCBI"):
        if not gene_names:
            st.info("Annotation GFF3 d'abord pour obtenir des symboles de gènes.")
        else:
            info = fetch_ncbi_gene_info(gene_names[:20], cfg["taxon"])
            if not info.empty:
                st.dataframe(info, use_container_width=True)

    # --- AlphaFold
    st.subheader("Structure 3D (AlphaFold)")
    gene = st.text_input("Nom du gène")
    if st.button("Afficher structure AlphaFold") and gene:
        acc, src = fetch_uniprot_id(gene, cfg["taxon"])
        if not acc:
            st.error(f"Aucun identifiant UniProt trouvé pour `{gene}`.")
        else:
            pdb = fetch_alphafold_pdb(acc, debug=True)
            if pdb is None:
                st.warning(f"AlphaFold ne propose pas de modèle pour `{acc}` ({src}). "
                           f"Vérifiez sur https://alphafold.ebi.ac.uk/entry/{acc}")
            else:
                st.caption(f"Structure AlphaFold pour `{acc}` ({src})")
                render_3d_structure(pdb)

    # --- STRING
    st.subheader("Réseau STRING")
    if st.button("Construire réseau STRING"):
        if len(gene_names) < 2:
            st.info("Il faut au moins 2 gènes annotés (GFF3) pour STRING.")
        else:
            species_id = "9838" if "Camelin" in st.session_state.species else "9913"
            edges, nodes = query_string_network(gene_names[:20], species_id)
            if not edges.empty:
                fig = plot_string_network(edges, nodes)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

def _tab_export():
    st.header("📤 Export")
    if not _require_data():
        return
    gt, ind_df, snp_df = st.session_state.gt, st.session_state.ind, st.session_state.snp

    st.subheader("PLINK binary")
    if st.button("Préparer .bed/.bim/.fam"):
        try:
            z = build_plink(gt, ind_df, snp_df, prefix="genome")
            st.download_button("⬇️ Télécharger PLINK.zip", z,
                               file_name="genome.zip", mime="application/zip")
        except Exception as e:
            st.error(f"❌ {e}")

    st.subheader("VCF")
    if st.button("Préparer VCF"):
        try:
            vcf = build_vcf(gt, ind_df, snp_df, proj="Pipeline v23.9")
            st.download_button("⬇️ Télécharger .vcf", vcf,
                               file_name="genome.vcf", mime="text/plain")
        except Exception as e:
            st.error(f"❌ {e}")

    st.subheader("CSV des résultats")
    for key, fname in (("gwas_res", "gwas.csv"),
                       ("diversity", "diversity.csv"),
                       ("froh", "froh.csv"),
                       ("tajima", "tajima.csv")):
        obj = st.session_state.get(key)
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            st.download_button(f"⬇️ {fname}", obj.to_csv(index=False),
                               file_name=fname, mime="text/csv")

def main():
    _init_state()
    st.title("🧬 Universal Livestock Genomics Pipeline v23.9")
    st.caption("Bovin & Camelin — PED/MAP/VCF, QC, structure, GWAS, annotation, export")

    _sidebar()

    tabs = st.tabs([
        "📁 Import", "🧹 QC", "🧬 Structure", "📊 Diversité & Fst",
        "🔗 LD / Ne / ROH", "🎯 GWAS", "💾 GBLUP", "🔬 Annotation",
        "📤 Export",
    ])
    with tabs[0]: _tab_import()
    with tabs[1]: _tab_qc()
    with tabs[2]: _tab_structure()
    with tabs[3]: _tab_diversity()
    with tabs[4]: _tab_ld()
    with tabs[5]: _tab_gwas()
    with tabs[6]: _tab_gblup()
    with tabs[7]: _tab_annotation()
    with tabs[8]: _tab_export()

if __name__ == "__main__":
    main()
