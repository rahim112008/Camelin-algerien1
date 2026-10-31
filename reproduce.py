# %% [markdown]
# # 🧬 Universal Genomics Pipeline v23.5 — Reproduction Complète
#
# **Objectif** : reproduire toutes les analyses publiées sur données simulées
# (bovin), avec valeurs numériques et figures sauvegardées dans `results/`.
#
# **Auteur** : Pipeline v23.5 (édition publication)
# **Durée** : ~10-15 minutes sur laptop standard (200 ind × 2000 SNPs)
#
# **Sorties** :
# - `results/summary.json` — métriques globales
# - `results/*.csv` — tableaux numériques
# - `results/*.html` — figures Plotly interactives
# - `results/*.png` — figures statiques (via kaleido)
# - `results/*.npy` — matrices (KING, FST, CI)
# - `results/14_bovine.zip/.vcf` — exports
#
# **Références clés** :
# - Weir & Cockerham 1984, Evolution 38:1358
# - Wigginton 2005, AJHG 76:887
# - Tajima 1989, Genetics 123:585
# - Reynolds 1983, Genetics 105:767
# - Botstein 1980, AJHG 32:314
# - Sved 1971, Theor Popul Biol 2:125
# - Wakefield 2009, Genet Epidemiol 33:79
# - VanRaden 2008, J Dairy Sci 91:4414
# - Randhawa 2014, Front Genet 5:295
# - Kang 2010, Nat Genet 42:348
# - Manichaikul 2010, Bioinformatics 26:2867

# %% [markdown]
# ## 0. Setup : mock Streamlit + import du module
#
# Le pipeline principal utilise `streamlit` pour l'UI. On désactive le cache
# et les appels UI pour pouvoir importer les fonctions pures dans un contexte
# script/notebook.

# %%
import warnings, sys, os, json, hashlib, platform
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Mock Streamlit AVANT import du module ---
import streamlit as st

def _no_cache(*args, **kwargs):
    """No-op cache : décorateur transparent."""
    if args and callable(args[0]):
        return args[0]
    return lambda f: f

st.cache_data = _no_cache
st.cache_resource = _no_cache
st.cache = _no_cache
st.set_page_config = lambda *a, **kw: None
for _ui in ("title", "caption", "markdown", "text", "write", "warning",
            "info", "error", "success", "header", "subheader", "divider"):
    if hasattr(st, _ui):
        setattr(st, _ui, lambda *a, **kw: None)

# --- Imports externes ---
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

# --- Créer le dossier résultats ---
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
print(f"✅ Dossier résultats : {RESULTS.resolve()}")

# --- Importer le module principal ---
sys.path.insert(0, ".")
try:
    import genomics_v23_5 as gp
except ImportError as e:
    raise SystemExit(f"❌ Placer 'genomics_v23_5.py' dans {Path.cwd()}\n{e}")

print(f"✅ Module importé : {gp.__name__}")
print(f"   Espèces disponibles : {list(gp.SPECIES_CONFIG.keys())}")

# %% [markdown]
# ## 1. Génération des données simulées (bovin)
#
# Simulation de 200 individus répartis en 4 populations, 2000 SNPs.
# Les génotypes sont générés selon un modèle Hardy-Weinberg avec
# différenciation entre populations (β distribution + drift).

# %%
CFG = gp.SPECIES_CONFIG["Bovin (Taurin/Zébu)"]
AUTO = CFG["autosomes"]
ARS = CFG["ars"]

gt, ind_df, snp_df = gp.gen_demo(
    n_ind=200, n_snp=2000, n_pop=4, seed=42,
    autosomes=AUTO, ars=ARS)
ind_df = gp.rec_fid(ind_df)

print(f"📊 Données simulées : {gt.shape[0]} individus × {gt.shape[1]} SNPs")
print(f"   Populations : {sorted(ind_df['FID'].unique())}")
print(f"   Missing rate : {np.isnan(gt).mean():.4f}")

# Hash pour reproductibilité
data_hash = hashlib.sha256(gt.tobytes() + snp_df.to_json().encode()).hexdigest()[:16]
print(f"   SHA-256 (16 premiers) : {data_hash}")

# Sauvegarder les données brutes pour référence
np.save(RESULTS / "00_genotypes_raw.npy", gt)
snp_df.to_csv(RESULTS / "00_snp_map.csv", index=False)
ind_df.to_csv(RESULTS / "00_individuals.csv", index=False)

# %% [markdown]
# ## 2. Contrôle Qualité (QC)
#
# Filtres standards (Anderson et al. 2010, PLoS One 5:e11542) :
# - Missing SNP < 5%
# - Missing individu < 5%
# - MAF ≥ 5%
# - HWE exact p ≥ 1e-6 (Wigginton 2005)
# - Exclure hétérozygotie ±3σ par population

# %%
params = {"geno": 0.05, "mind": 0.05, "maf": 0.05, "hwe": 1e-6, "het_sd": 3.0}

gt_filt, ind_filt, snp_filt, gt_pre, ind_pre, snp_pre, qc_stats = gp.apply_qc(
    gt, ind_df, snp_df, params, AUTO, keep_autosomes=True, rem_out=True)

with open(RESULTS / "01_qc_metrics.json", "w") as f:
    json.dump(qc_stats, f, indent=2)

print(f"📊 Après QC : {gt_filt.shape[0]} ind × {gt_filt.shape[1]} SNPs")
print(f"   Individus exclus : {qc_stats['excluded_ind']}")
print(f"   SNPs exclus : {qc_stats['excluded_snp']}")

# --- LD pruning ---
keep = gp.ld_pruning(gt_filt, w=50, step=5, r2=0.2)
gt_pruned = gt_filt[:, keep]
snp_pruned = snp_filt[keep].reset_index(drop=True)

print(f"📊 Après LD pruning : {gt_pruned.shape[1]} SNPs")
print(f"   Réduction : {100*(1-gt_pruned.shape[1]/gt_filt.shape[1]):.1f}%")

np.save(RESULTS / "01_genotypes_qc.npy", gt_filt)
np.save(RESULTS / "01_genotypes_pruned.npy", gt_pruned)
snp_pruned.to_csv(RESULTS / "01_snp_map_pruned.csv", index=False)

# %% [markdown]
# ## 3. Indices de Diversité Génétique
#
# **He** (Nei 1973), **Ho**, **Fis**, **PIC** (Botstein 1980),
# **MAF**, nombre de SNPs polymorphes.

# %%
labels = ind_filt["FID"].astype(str).values
div_df = gp.compute_diversity(gt_pruned, labels, data_hash)

div_df.to_csv(RESULTS / "02_diversity.csv", index=False)
print("\n📊 Indices de diversité :")
print(div_df.to_string(index=False))

# Figure
fig_div = px.bar(div_df, x="Population",
                  y=["H_Observee (Ho)", "H_Attendue (He)", "PIC"],
                  barmode="group",
                  title="Diversité génétique par population (Nei 1973, Botstein 1980)")
fig_div.write_html(RESULTS / "02_diversity.html")
try:
    fig_div.write_image(RESULTS / "02_diversity.png", width=1000, height=500)
except Exception:
    pass

# %% [markdown]
# ## 4. Structure des populations (PCA + KING)
#
# **PCA** (Patterson 2006) sur génotypes centrés.
# **KING-robust** (Manichaikul 2010) pour la matrice de relatedness.

# %%
pca_s, pca_v = gp.pca_analysis(gt_pruned, nc=10)
king_K = gp.king_mat(gt_pruned)

pca_df = pd.DataFrame({
    "IID": ind_filt["IID"].values, "FID": labels,
    **{f"PC{i+1}": pca_s[:, i] for i in range(5)}
})
pca_df.to_csv(RESULTS / "03_pca.csv", index=False)
np.save(RESULTS / "03_king.npy", king_K)

print(f"📊 PCA : PC1={pca_v[0]:.2f}%, PC2={pca_v[1]:.2f}%, "
      f"PC3={pca_v[2]:.2f}%, PC4={pca_v[3]:.2f}%, PC5={pca_v[4]:.2f}%")

fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="FID",
                      title=f"PCA — PC1({pca_v[0]:.1f}%) vs PC2({pca_v[1]:.1f}%)")
fig_pca.write_html(RESULTS / "03_pca.html")
try:
    fig_pca.write_image(RESULTS / "03_pca.png", width=900, height=600)
except Exception:
    pass

# %% [markdown]
# ## 5. FST pairwise — Weir & Cockerham (1984) avec bootstrap CI
#
# **WC84** (Evolution 38:1358) avec **block bootstrap** (Efron 1979)
# pour IC 95% — indispensable pour publication.

# %%
fst_M, pops = gp.fst_pairwise(gt_pruned, labels, use_wc84=True)
np.save(RESULTS / "05_fst_pairwise.npy", fst_M)
pd.DataFrame(fst_M, index=pops, columns=pops).to_csv(
    RESULTS / "05_fst_pairwise.csv")
print(f"📊 FST pairwise ({len(pops)} populations) :")
print(pd.DataFrame(fst_M, index=pops, columns=pops).round(4).to_string())

# --- Bootstrap CI ---
print("\n⏳ Bootstrap 100 rééchantillonnages (block=50 SNPs)...")
fst_M_boot, ci_lo, ci_hi, pops_boot = gp.fst_pairwise_bootstrap(
    gt_pruned, labels, n_boot=100, block=50, use_wc84=True)
np.save(RESULTS / "05_fst_ci_lo.npy", ci_lo)
np.save(RESULTS / "05_fst_ci_hi.npy", ci_hi)

# Tableau avec IC
K = len(pops)
rows = []
for i in range(K):
    for j in range(i+1, K):
        rows.append({
            "Pop_A": pops[i], "Pop_B": pops[j],
            "FST": round(fst_M[i, j], 4),
            "CI_lo": round(ci_lo[i, j], 4),
            "CI_hi": round(ci_hi[i, j], 4)
        })
ci_df = pd.DataFrame(rows)
ci_df.to_csv(RESULTS / "05_fst_pairwise_CI.csv", index=False)
print("\n📊 FST avec IC 95% :")
print(ci_df.to_string(index=False))

# Heatmap
fig_fst = px.imshow(fst_M, x=pops, y=pops, color_continuous_scale="Viridis",
                     title="FST pairwise — Weir & Cockerham 1984",
                     aspect="auto", text_auto=".3f")
fig_fst.write_html(RESULTS / "05_fst_heatmap.html")
try:
    fig_fst.write_image(RESULTS / "05_fst_heatmap.png", width=700, height=600)
except Exception:
    pass

# %% [markdown]
# ## 6. Arbre UPGMA — Reynolds (1983) avec support bootstrap
#
# Distance de Reynolds `D = −log(1 − FST)`, UPGMA (average linkage),
# support par 100 bootstraps sur SNPs (Felsenstein 1985).

# %%
print("⏳ UPGMA + 100 bootstraps...")
upgma_res = gp.upgma_bootstrap(gt_pruned, snp_pruned, labels, n_boot=100, seed=42)

with open(RESULTS / "06_upgma_newick.txt", "w") as f:
    f.write(upgma_res["newick"])
np.savetxt(RESULTS / "06_upgma_support.txt", upgma_res["support"], fmt="%.0f")

print(f"\n🌳 Newick (avec support %) :")
print(upgma_res["newick"])
print(f"\n   Supports : {upgma_res['support'].tolist()}")
print(f"   Nœuds avec support ≥70% : {(upgma_res['support']>=70).sum()}/{len(upgma_res['support'])}")

# %% [markdown]
# ## 7. Décroissance du LD

# %%
ld_df = gp.ld_decay(gt_pruned, snp_pruned, max_kb=1000, max_snp=3000)
ld_df.to_csv(RESULTS / "07_ld_decay.csv", index=False)
print(f"📊 LD decay : {len(ld_df)} bins")

if len(ld_df) > 0:
    fig_ld = px.line(ld_df, x="dist_kb", y="r2",
                      title="Décroissance du LD par distance physique")
    fig_ld.write_html(RESULTS / "07_ld_decay.html")
    try:
        fig_ld.write_image(RESULTS / "07_ld_decay.png", width=900, height=500)
    except Exception:
        pass

# %% [markdown]
# ## 8. Tajima's D — Signature de sélection
#
# Test de neutralité (Tajima 1989, Genetics 123:585) par fenêtres
# glissantes sur une population donnée.

# %%
pop_test = pops[0]
mask = labels == pop_test

if mask.sum() >= 5:
    print(f"⏳ Tajima's D sur {pop_test} ({mask.sum()} ind)...")
    taj = gp.tajimas_scan(gt_pruned[mask], snp_pruned, w=25, step=10)
    taj.to_csv(RESULTS / "08_tajima.csv", index=False)

    if not taj.empty:
        print(f"   {len(taj)} fenêtres")
        print(f"   Tajima's D min = {taj['TAJIMA_D'].min():.3f}")
        print(f"   Tajima's D max = {taj['TAJIMA_D'].max():.3f}")
        print(f"   Tajima's D moyen = {taj['TAJIMA_D'].mean():.3f}")

        fig_taj = px.scatter(taj, x="MID", y="TAJIMA_D", color="CHR",
                              title=f"Tajima's D — {pop_test} (fenêtres 25 SNPs)")
        fig_taj.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_taj.write_html(RESULTS / "08_tajima.html")
        try:
            fig_taj.write_image(RESULTS / "08_tajima.png", width=1100, height=500)
        except Exception:
            pass
else:
    print(f"⚠️ Population {pop_test} a < 5 ind, Tajima's D ignoré")

# %% [markdown]
# ## 9. Composite Selection Score (CSS)
#
# **CSS** = |Z(FST)| × |Z(HOM)| (Randhawa 2014, Front Genet 5:295).

# %%
css_df = gp.composite_score(gt_pruned, snp_pruned, labels)
css_df.to_csv(RESULTS / "09_css.csv", index=False)

print(f"📊 CSS : {len(css_df)} SNPs")
print("\n   Top 10 SNPs :")
print(css_df.head(10)[["SNP", "CHR", "BP", "FST", "HOM", "CSS"]].to_string(index=False))

# %% [markdown]
# ## 10. GWAS — LMM EMMAX-like
#
# Pheontype simulé (héritable) : trait = 0.5 × 3 SNPs causaux + bruit.
# Modèle : EMMAX-like (Kang et al. 2010) avec GRM + 5 PCs.

# %%
rng = np.random.default_rng(42)
n = gt_pruned.shape[0]
pheno = rng.normal(0, 1, n)

# Ajouter signal génétique
for s in [0, 100, 500]:
    if s < gt_pruned.shape[1]:
        pheno += gt_pruned[:, s] * 0.5

# 10% NaN pour simuler données réelles
pheno[rng.random(n) < 0.10] = np.nan
pd.DataFrame({"IID": ind_filt["IID"].values, "FID": labels,
              "trait": pheno}).to_csv(RESULTS / "10_phenotype.csv", index=False)

print(f"📊 Phénotype simulé : {np.isfinite(pheno).sum()} valeurs valides")
print("⏳ GWAS LMM EMMAX-like...")

r = gp.run_gwas(gt_pruned, pheno, king_K, pca_s[:, :5], is_bin=False)
r = gp.annotate_gwas(r, snp_pruned)
r = gp.postprocess_gwas(r, apply_gc=True)
r.to_csv(RESULTS / "10_gwas.csv", index=False)

lam_gc = gp.lambda_gc(r["P"].values)
n_bonf = int((r["P"] < 0.05/len(r)).sum())
n_fdr = int((r.get("FDR", pd.Series([1]*len(r))) < 0.05).sum()) if "FDR" in r.columns else 0

print(f"\n📊 GWAS : {len(r)} SNPs testés")
print(f"   λGC = {lam_gc:.3f}")
print(f"   Bonferroni<0.05 : {n_bonf} SNPs")
print(f"   FDR<0.05 : {n_fdr} SNPs")

if not r.empty:
    print(f"\n   Top 5 SNPs :")
    cols = [c for c in ["SNP", "CHR", "BP", "MAF", "BETA", "P"] if c in r.columns]
    print(r[cols].head(5).to_string(index=False))

# Manhattan
fig_manh = px.scatter(r, x="BP", y=r["log10P"], color="CHR",
                       hover_data=["SNP", "P"],
                       title=f"Manhattan — λGC = {lam_gc:.3f}")
fig_manh.add_hline(y=-np.log10(0.05/len(r)), line_dash="dash", line_color="red")
fig_manh.write_html(RESULTS / "10_gwas_manhattan.html")
try:
    fig_manh.write_image(RESULTS / "10_gwas_manhattan.png", width=1100, height=500)
except Exception:
    pass

# QQ plot
fig_qq = px.scatter(
    x=-np.log10(np.sort(r["P"].values)),
    y=-np.log10(np.arange(1, len(r)+1) / (len(r)+1)),
    labels={"x": "Attendu (-log10 P)", "y": "Observé (-log10 P)"},
    title="QQ-plot GWAS")
fig_qq.add_shape(type="line", x0=0, y0=0,
                  x1=max(-np.log10(r["P"].min()), 1),
                  y1=max(-np.log10(r["P"].min()), 1),
                  line=dict(dash="dash", color="red"))
fig_qq.write_html(RESULTS / "10_gwas_qq.html")

# %% [markdown]
# ## 11. Fine-mapping — ABF Wakefield 2009
#
# `ABF = √(1−W) · exp(BETA²·W / (2·SE²))` avec `W = v / (SE² + v)`
# où `v` = variance a priori (0.04 par défaut, Wakefield 2009).

# %%
pip = gp.fine_mapping(r, prior_var=0.04)
pip.to_csv(RESULTS / "11_fine_mapping_pip.csv", index=False)

print("📊 Fine-mapping PIP — Top 10 :")
cols_pip = [c for c in ["SNP", "CHR", "BP", "P", "BETA", "SE", "PIP"] if c in pip.columns]
print(pip[cols_pip].head(10).to_string(index=False))

print(f"\n   SNPs avec PIP > 0.5 : {(pip['PIP'] > 0.5).sum()}")
print(f"   SNPs avec PIP > 0.9 : {(pip['PIP'] > 0.9).sum()}")

fig_pip = px.bar(pip.head(20), x="SNP", y="PIP", color="PIP",
                  color_continuous_scale="Viridis",
                  title="Posterior Inclusion Probability (Wakefield 2009)")
fig_pip.write_html(RESULTS / "11_fine_mapping_pip.html")
try:
    fig_pip.write_image(RESULTS / "11_fine_mapping_pip.png", width=1000, height=500)
except Exception:
    pass

# %% [markdown]
# ## 12. GBLUP — Héritabilité REML
#
# **GBLUP** (VanRaden 2008) avec **h² estimée par REML profile**.
# Validation croisée 5-fold pour le R² prédictif.

# %%
print("⏳ GBLUP + REML + CV 5-fold...")
gebv_df, gblup_stats = gp.run_gblup(gt_pruned, pheno, kf=5)

gebv_df.to_csv(RESULTS / "12_gblup_gebv.csv", index=False)
with open(RESULTS / "12_gblup_stats.json", "w") as f:
    json.dump(gblup_stats, f, indent=2)

print(f"\n📊 GBLUP :")
print(f"   h² (REML) : {gblup_stats['h2']:.3f}")
print(f"   CV r² : {gblup_stats['r2']:.3f}")
print(f"   CV corr : {gblup_stats['corr']:.3f}")
print(f"   RMSE : {gblup_stats['rmse']:.3f}")

fig_gblup = px.scatter(gebv_df, x="Phenotype", y="GEBV",
                        title=f"GBLUP — h² REML = {gblup_stats['h2']:.3f}, CV r² = {gblup_stats['r2']:.3f}")
fig_gblup.write_html(RESULTS / "12_gblup.html")
try:
    fig_gblup.write_image(RESULTS / "12_gblup.png", width=800, height=600)
except Exception:
    pass

# %% [markdown]
# ## 13. Exports PLINK + VCF
#
# Formats standards pour interopérabilité.

# %%
plink_zip = gp.build_plink(gt_pruned, ind_filt, snp_pruned, "bovine")
with open(RESULTS / "13_bovine_plink.zip", "wb") as f:
    f.write(plink_zip)
print(f"📦 PLINK : {len(plink_zip)/1024:.1f} KB")

vcf_str = gp.build_vcf(gt_pruned, ind_filt, snp_pruned, "bovine_v23.5")
with open(RESULTS / "13_bovine.vcf", "w") as f:
    f.write(vcf_str)
print(f"📄 VCF : {len(vcf_str)/1024:.1f} KB")

# %% [markdown]
# ## 14. Résumé pour publication
#
# Le fichier `summary.json` capture toutes les métriques clés
# pour inclusion directe dans le matériel supplémentaire.

# %%
summary = {
    "pipeline_version": "v23.5",
    "timestamp": datetime.now().isoformat(),
    "species": "Bovin (Taurin/Zébu)",
    "data_hash": data_hash,
    "n_ind_final": int(gt_filt.shape[0]),
    "n_snp_final": int(gt_filt.shape[1]),
    "n_snp_pruned": int(gt_pruned.shape[1]),
    "qc_excluded_ind": int(qc_stats["excluded_ind"]),
    "qc_excluded_snp": int(qc_stats["excluded_snp"]),
    "diversity": {
        "mean_He": float(div_df["H_Attendue (He)"].mean()),
        "mean_Ho": float(div_df["H_Observee (Ho)"].mean()),
        "mean_PIC": float(div_df["PIC"].mean()),
        "mean_Fis": float(div_df["Fis (Consanguinité)"].mean()),
    },
    "pca_variance_pct": [float(x) for x in pca_v[:5]],
    "fst_pairwise": {
        "mean": float(np.nanmean(fst_M[np.triu_indices_from(fst_M, k=1)])),
        "max": float(np.nanmax(fst_M[np.triu_indices_from(fst_M, k=1)])),
        "min": float(np.nanmin(fst_M[np.triu_indices_from(fst_M, k=1)])),
    },
    "upgma_support_ge70": int((upgma_res["support"] >= 70).sum()),
    "upgma_total_nodes": int(len(upgma_res["support"])),
    "gwas": {
        "n_snp_tested": int(len(r)),
        "lambda_gc": float(lam_gc),
        "n_bonferroni_05": int(n_bonf),
        "n_fdr_05": int(n_fdr),
    },
    "fine_mapping": {
        "n_pip_gt_05": int((pip["PIP"] > 0.5).sum()),
        "n_pip_gt_09": int((pip["PIP"] > 0.9).sum()),
    },
    "gblup": {
        "h2_reml": float(gblup_stats["h2"]),
        "cv_r2": float(gblup_stats["r2"]),
        "cv_corr": float(gblup_stats["corr"]),
        "rmse": float(gblup_stats["rmse"]),
    },
    "software": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    },
}

with open(RESULTS / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("📊 RÉSUMÉ FINAL POUR PUBLICATION")
print("=" * 60)
print(json.dumps(summary, indent=2))

print("\n" + "=" * 60)
print(f"✅ TOUS LES RÉSULTATS SONT DANS : {RESULTS.resolve()}")
print("=" * 60)

# %% [markdown]
# ## ✅ Reproduction terminée
#
# **Fichiers générés** (dans `results/`) :
#
# | Fichier | Contenu |
# |---------|---------|
# | `00_*.npy` / `00_*.csv` | Données brutes simulées |
# | `01_qc_metrics.json` | Statistiques QC |
# | `02_diversity.csv` / `.html` | Indices de diversité |
# | `03_pca.csv` / `.html` | PCA + matrice KING |
# | `05_fst_pairwise*.npy` / `.csv` / `.html` | FST + IC 95% |
# | `06_upgma_newick.txt` / `_support.txt` | Arbre UPGMA |
# | `07_ld_decay.csv` / `.html` | Décroissance LD |
# | `08_tajima.csv` / `.html` | Tajima's D |
# | `09_css.csv` | Composite Selection Score |
# | `10_gwas.csv` / `_manhattan.html` / `_qq.html` | GWAS |
# | `11_fine_mapping_pip.csv` / `.html` | Fine-mapping ABF |
# | `12_gblup_gebv.csv` / `_stats.json` | GBLUP + h² REML |
# | `13_bovine_plink.zip` / `13_bovine.vcf` | Exports |
# | **`summary.json`** | **Métriques globales (pour publication)** |
#
# **Prochaine étape** : copier `results/summary.json` dans le matériel supplémentaire
# de votre publication, et citer ce notebook comme code de reproduction.
