"""
For every sample and every ordered pair of cell types (sender S -> receiver T) it scores each
ligand->receptor pair of CellPhoneDB as (mean ligand in S + mean receptor in T) / 2, tests it against a null
built by shuffling the cell-type labels, keeps the pairs whose ligand AND receptor are specifically enriched in
their own cell type, requires the pair to replicate across the samples of a condition, and finally draws the
resulting sender -> receiver network.

"""


##### some comments are AI generated (including matrices) in order to have a better and faster representation of some data and commands.




# =========================================================================================================
# CLI (cmd_run)
#
# Argomento            | Default  | Significato Operativo
# ---------------------------------------------------------------------------------------------------------
# --adata              | Required | Percorso del file .h5ad con i dati di espressione log-normalizzati
# --cpdb-dir           | Required | Cartella con i file CSV del database CellPhoneDB (v5.0.0)
# --sample-col         | None     | Colonna/e obs che definiscono il singolo campione/paziente (replicato)
# --condition-col      | None     | Colonna obs che definisce la condizione (es. CRC vs CRLM vs PBMC)
# --n-perm             | 1000     | Numero di permutazioni per calcolare il p-value empirico
# --correction         | 'none'   | Correzione per test multipli ('none' o 'fdr_bh')
# --require-specific   | True     | Richiede che sia il ligando che il recettore siano significativi
# --min-sample-frac    | 1.0      | Frazione minima di campioni idonei in cui l'interazione dev'essere sig.
# --negative-control   | False    | Shuffla le etichette per verificare il controllo dei falsi positivi
# =========================================================================================================


import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import logging
import re
import sys
import time
import zipfile
from collections import deque
from dataclasses import dataclass

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

__version__ = "1.0.0"

log = logging.getLogger("CellTalkPy")


def _read_table(cpdb_dir, name):
    ''' read the "name" file from the Cpdb directory or from cellphonedb.zip'''
    loose = os.path.join(cpdb_dir, name)
    if os.path.exists(loose):
        return pd.read_csv(loose)
    zpath = os.path.join(cpdb_dir, "cellphonedb.zip")
    if os.path.exists(zpath): #if zipped
        with zipfile.ZipFile(zpath) as z:
            for member in z.namelist():
                if os.path.basename(member)==name:
                    with z.open(member) as fh:
                        return pd.read_csv(fh)
    raise FileNotFoundError(f'Cannot find the {name} file in the Cpdb directory {cpdb_dir}')


def _to_bool(s):
    '''csv boolenas can be TRUE/ "TRUE"/ 1/yes --> this function normalize everything to 1'''
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

def _clean(v):
    """Blank CSV cells are NaN; make them '' (NaN keys would be silently dropped by pandas groupby)."""
    return "" if pd.isna(v) else str(v)

def _strip_version(x):
    """ENSG00000123456.12 -> ENSG00000123456 (data often carry the version suffix)."""
    return str(x).split(".")[0]

###### class CpdbTables:

"""questa classe serve a raggruppare e tipizzare in un unico oggetto 
le tabelle del database CellPhoneDB caricate in memoria tramite Pandas, facilitandone il passaggio tra le varie funzioni dell'applicazion"""
@dataclass
class CpdbTables:
    genes: pd.DataFrame
    proteins: pd.DataFrame
    complexes: pd.DataFrame
    interactions: pd.DataFrame



REQUIRED_TABLES= ["gene_input.csv", "protein_input.csv", "complex_input.csv", "interaction_input.csv"]
def load_cpdb(cpdb_dir):
    """load the four cellphonedb tables"""
    t =CpdbTables(*[_read_table(cpdb_dir, n) for n in REQUIRED_TABLES])
    #definying the needed dictionary with the necessary columns, if at least one column is missing --> value error
    need = {"gene_input": (t.genes, ["uniprot"]), "protein_input": (t.proteins, ["uniprot"]),
            "complex_input": (t.complexes, ['complex_name']),
            "interaction_input": (t.interactions, ['partner_a', 'partner_b'])}
    for name, (df, cols) in need.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f'Missing columns {missing} in {name} table')

        # print a log message with the counts of rows for each table (number of genes, proteins, complexes, and interaction loaded)
        log.info("CellPhoneDB loaded: %d genes, %d proteins, %d complexes, %d interactions.",
                 len(t.genes), len(t.proteins), len(t.complexes), len(t.interactions))
    return t



@dataclass
class Model:
    """the needed statistics already converted to arrays.
    -genes : the adata columns used (G) (it's a list)

    - A : sparse (P x G) matrix. (from genes to proteins)
    --> if the protein is made by gene_A --> protein_1 = Gene_A
    --> if the protein is made by gene_B and gene_C --> protein_2 = mean of the 2 genes

    -sub_idx (from protein to complexes): in complexes it looks for the lowest expressed protein.
    - partners: partners ids (uniprot or complex name) with length Q.
    """

    genes: list
    A: sp.csr_matrix #scipy sparse amtrix
    sub_idx: np.ndarray
    partners: list
    inter: pd.DataFrame




def build_model(tables, var_names, gene_id_type='symbol'):
    """filtering the database with the actual data and return the Model"""
    var_names= pd.Index(var_names)

    # ---- 1.1 uniprot -> gene(s) present in the data --------------------------------------------
    # 1)it keeps the gene list (both gene symbol and ensembl) and search within the database which are the corresponding proteins,
    # 2)remove from the database all the non-necessary protein
    #3) order and indexing genes from 0 to G-1, and proteins from 0 to P-1

    g = tables.genes.copy()
    if gene_id_type == "symbol":
        parts = [g[["uniprot", c]].rename(columns={c: "key"}) for c in ("hgnc_symbol", "gene_name") if c in g.columns] # if 'gene symbol' --> the function joins both the hgnc_symbol and gene_name database's columns in a single column named 'KEY'
        keys = pd.concat(parts).dropna()
        lookup = pd.Series(var_names, index=var_names)   #lookup is a series where both the indexes (cleaned name) and values (real gene names)are theb correspondence to var_names                    # key -> var name
    elif gene_id_type == "ensembl":
        if "ensembl" not in g.columns:
            raise ValueError("gene_input.csv has no 'ensembl' column: use --gene-id-type symbol")
        keys = g[["uniprot", "ensembl"]].rename(columns={"ensembl": "key"}).dropna()
        keys["key"] = keys["key"].map(_strip_version)
        lookup = pd.Series(var_names, index=var_names.map(_strip_version))
    else:
        raise ValueError("gene_id_type must be 'symbol' or 'ensembl'")


    lookup = lookup[~lookup.index.duplicated()]
    keys = keys[keys["key"].isin(lookup.index)].drop_duplicates()
    keys["var"] = keys["key"].map(lookup) # --> lookup looks for the correspondence e returns the exact name used
    keys = keys[["uniprot", "var"]].drop_duplicates()
    if keys.empty:
        raise ValueError("No gene of the database matches the gene names of the data. Check --gene-col / "
                         f"--gene-id-type (example names in the data: {list(var_names[:3])}).")

    genes = sorted(keys["var"].unique()) #sort the gene names
    gene_pos = {x: i for i, x in enumerate(genes)} #gene postions
    proteins = sorted(keys["uniprot"].unique()) #sorting the protein names
    prot_pos = {u: i for i, u in enumerate(proteins)} #protein positioins
    P, G = len(proteins), len(genes)
    log.info("%d database proteins are encoded by %d genes present in the data.", P, G)





    # ---- 1.2 sparse averaging matrix A (proteins x genes) ---------------------------------------
    # sparse matrix A --> it builds the matrix A = PxG in order to convert the gene expression in protein expression
    # if a protein is associated to N different genes it assigns a weight of 1/N to each gene
    #sparse matrix in order to not waste memory (majority of 0s)
    n_genes_of = keys.groupby("uniprot")["var"].nunique() #number of genes
    rows = keys["uniprot"].map(prot_pos).to_numpy()
    cols = keys["var"].map(gene_pos).to_numpy()
    vals = (1.0 / keys["uniprot"].map(n_genes_of)).to_numpy(dtype=np.float32)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(P, G)) #the actual sparse matrix (PxG) --> if the protein is translated from just 1 gene the value will be 1, if more genes 1/N_genes




    # ---- 1.3 partners: proteins and complexes ----------------------------------------------------
    # it identifies all the partners in the database and build the index matrix of sub-units --> sub_idx
   # remember that prot_pos is {'prot_1': 0, 'prot_2':1, ... ecc}



    cx = tables.complexes #complex table (csv)
    uni_cols = [c for c in cx.columns if c.startswith("uniprot_")] # unirpto columns in the table -->  uniprot_1 ... uniprot_N
    complex_subunits = {}
    #creates a dict (k,v)-->(complex_subunit) where the keys  are the complex name and the values are the uniprot names (for example IL35: ['Q14213', 'P29459'])
    for _, row in cx.iterrows():
        subs = [str(row[c]).strip() for c in uni_cols if pd.notna(row[c]) and str(row[c]).strip()]
        complex_subunits[row["complex_name"]] = subs



    receptor = {} # partner -> is it a receptor?
    if "receptor" in tables.proteins.columns: #it is a columns within the protein table named 'receptor'
        receptor.update(dict(zip(tables.proteins["uniprot"], _to_bool(tables.proteins["receptor"])))) #create a dictionary with boolean values (True or False) that says if a molecole is a receptor or not within the protein table
    if "receptor" in cx.columns: #the same with the complex table
        receptor.update(dict(zip(cx["complex_name"], _to_bool(cx["receptor"]))))


    sym_col = "hgnc_symbol" if "hgnc_symbol" in tables.genes.columns else "gene_name"
    label_of = dict(zip(tables.genes["uniprot"], tables.genes[sym_col]))  #creates a dictionary of 'translation' --> from Uniprot to gene_symbol (for example 'P05231' --> 'IL6')

    inter_raw = tables.interactions
    all_partners = pd.unique(pd.concat([inter_raw["partner_a"], inter_raw["partner_b"]]).dropna()) #join the two columns in a unique list and remove duplicates and null values -> at the end it returns the whole list of all the molecules/complexes envolved in interactions



    partners, sub_lists = [], []
    #for each partner of database it looks for 'p' (that is the current partner): if p is a complex it return the list of its subunits, if not it will a list of one-dimension
    for p in all_partners:
        subs = complex_subunits.get(p, [p]) # protein = complex of size 1
        if subs and all(s in prot_pos for s in subs): # ALL subunits need a gene in the data --> each subunit 's' belonging to 'p' is present in our data (so, registered in 'prot_pos' created before)--> if at least 1 subunit of the complex is abset in our data the whole 'p' partener/protein is discarded
            partners.append(p)
            sub_lists.append([prot_pos[s] for s in subs])
    if not partners:
        raise ValueError("No database partner can be evaluated with this dataset (check gene names).")

    # S computes the max dimension of all the partners (if the biggest complex has 3 subunits then S will be 3)
    S = max(len(s) for s in sub_lists)
    sub_idx = np.full((len(partners), S), P, dtype=np.int64) #--> numpy matrix with rows (number of valid partners) and columns (max dimension S). It will filled up by the pad value = P (+inf)
    for q, subs in enumerate(sub_lists): #for each valid partner (with its index 'q')
        sub_idx[q, :len(subs)] = subs
    partner_pos = {p: i for i, p in enumerate(partners)}
    log.info("%d / %d database partners can be evaluated with this dataset.", len(partners), len(all_partners))
    #




    # ---- 1.4 oriented interactions: who is the ligand, who the receptor? ------------------------
    # partner_a / partner_b of the database are NOT reliably (ligand, receptor). Rule:
    #   exactly one side flagged 'receptor'  -> that side is the receptor (1 orientation)
    #   otherwise (both flagged or none)     -> unknown direction: test BOTH orientations

    #so, it defines who is the ligand (lig) and receptor (rec) for each couple.
    # if the direction is certein (only one member is receptor) it defines the orientation. if the direction is uncertain it tests both the direction (a-->b, b-->a)
    records = []
    for idx, r in inter_raw.iterrows():
        a, b = r["partner_a"], r["partner_b"]
        if a not in partner_pos or b not in partner_pos:
            continue
        ra, rb = receptor.get(a, False), receptor.get(b, False)
        if ra and not rb:
            orient = [(b, a)]
        elif rb and not ra:
            orient = [(a, b)]
        else:
            orient = [(a, b)] if a == b else [(a, b), (b, a)]
        for lig, rec in orient:
            records.append({
                "interaction_id": idx, "lig": partner_pos[lig], "rec": partner_pos[rec],
                "ligand": lig, "receptor": rec,
                "ligand_label": label_of.get(lig, lig), "receptor_label": label_of.get(rec, rec),
                "directionality": _clean(r.get("directionality", "")),
                "classification": _clean(r.get("classification", ""))})
    inter = pd.DataFrame.from_records(records).drop_duplicates(subset=["lig", "rec"]).reset_index(drop=True)
    if inter.empty:
        raise ValueError("No interaction is evaluable with this dataset (check gene names).")
    log.info("%d oriented ligand->receptor pairs will be tested.", len(inter))
    return Model(genes=genes, A=A, sub_idx=sub_idx, partners=partners, inter=inter)
    #Model(- genes--> list of genes used in my data to find the protein in the database,
            # - A --> sparse matrix . usefull to convert i gene expression in protein expression (A x X where X is the gene expression matrix).
            # -sub_idx --> numerical grid useful for complex protein
            # - partner --> valid protein and complexes that will be used
            # - inter --> interaction table Ligand--> receptor












# 2. READING THE ANNDATA, SAMPLES AND CONDITIONS

# sample : the biological replicate analysed independently. It is defined by one or MORE obs columns that are concatenated: 'Patient' alone is not a replicate if a patient has several tissues,
# but 'Patient' + 'Tissue' is.
# e.g.  --sample-col Patient Tumor_type_recurrence --> it keeps all the columns together.

# condition : optional obs column grouping samples (e.g. CRC vs CRLM). Results are aggregated PER CONDITION.
# unit : one (condition, sample) block of cells = one permutation test.
#  No --sample-col -> each condition is one unit (its cells pooled). Neither -> one unit with all cells.


@dataclass
class Unit: #unit --> unit of analyses (group of cells)
    condition: str #experimental condition
    sample: str    # ID of the biological replicate (e.g. Patient_01|tumor_tissue)
    cell_idx: np.ndarray # row positions of the cells of this unit in the expression matrix


def _warn_if_raw_counts(X, n_check=20000):
    """ sanity check: log-normalised data have small, mostly non-integer values."""
    d = X.data[:n_check]
    if d.size and (d.max() > 30 or np.allclose(d, np.round(d))):
        log.warning("The expression matrix looks like RAW COUNTS (integers / large values). "
                    "Use normalised + log1p data (or pass --layer with the right one).")




def load_expression(adata, layer=None, use_raw=False, gene_col="index"):
    """it returns (X, gene_names)
    The matrix MUST be normalised, log1p-transformed expression"""
    if use_raw: #selected from user
        if adata.raw is None:
            raise ValueError("--use-raw was requested but adata.raw is None")
        X, var = adata.raw.X, adata.raw.var
    elif layer: #selected from user
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found. Available: {list(adata.layers.keys())}")
        X, var = adata.layers[layer], adata.var
    else: #if not seelcted it uses the actual .X
        X, var = adata.X, adata.var
    if gene_col != "index" and gene_col not in var.columns:
        raise ValueError(f"--gene-col '{gene_col}' not in adata.var columns: {list(var.columns)}")
    names = var.index if gene_col == "index" else var[gene_col]
    X = sp.csr_matrix(X, dtype=np.float32) # works for dense and sparse input
    _warn_if_raw_counts(X)
    return X, pd.Index(names.astype(str))



def make_units(obs, sample_cols, condition_col):
    """Split the cells into units (one unit for each condition)
    for example a patient with a specific tissue"""
    cond = pd.Series("all", index=obs.index) if not condition_col else obs[condition_col].astype(str)
    if sample_cols: #if the user inserts the sample col:
        for c in sample_cols:
            if c not in obs.columns:
                raise ValueError(f"--sample-col '{c}' not in adata.obs (columns: {list(obs.columns)})")
        sample = obs[list(sample_cols)].astype(str).agg("|".join, axis=1)# join of the columns with a '|' --> es: p01|CRLM
        n_cond = pd.DataFrame({"s": sample.values, "c": cond.values}).groupby("s")["c"].nunique()
        bad = n_cond[n_cond > 1].index.tolist()
        if bad:
            raise ValueError(f"Samples spanning more than one condition: {bad[:5]}. Add the condition column to "
                             f"--sample-col (e.g. --sample-col Patient {condition_col}).")
    else:
        sample = cond.copy()                                                       # pooled: sample == condition
    df = pd.DataFrame({"cond": cond.values, "sample": sample.values})
    units = [Unit(condition=c, sample=s, cell_idx=np.asarray(idx))
             for (c, s), idx in df.groupby(["cond", "sample"], sort=True).indices.items()]
    log.info("%d unit(s) (samples) to analyse across %d condition(s).", len(units), df["cond"].nunique())
    return units

















# 3. STATISTICS

# For ONE unit with K cell types:
#  1. mean expression mu[g,k] of every gene in every cell type, and fraction phi[g,k] of its cells with x > 0;
#  2. gene -> protein (average if several genes) -> partner (a COMPLEX is as strong as its weakest subunit:
#     MIN over subunits), for both mu and phi;
#  3. for an oriented interaction ligand L -> receptor R and ordered cell types (sender S, receiver T)
#         score = ( mu[L,S] + mu[R,T] ) / 2
#     TESTABLE only if phi[L,S] >= expr_frac and phi[R,T] >= expr_frac (and score > min_score);
#  4. the STATISTIC is the score if testable, 0 otherwise - defined identically on the real data and on every
#     permutation (otherwise the p-values would be anti-conservative: selecting tests by a real-data filter
#     picks those with an upward-fluctuating mean);
#  5. null: shuffle the cell-type labels among the cells of the unit (sizes preserved), recompute everything,
#     p = (1 + #{null >= real}) / (1 + n_perm)   [never 0; the smallest p is 1/(n_perm+1)];
#  6. the same permutations give partner-specificity p-values: p_ligand (is L enriched in S?) and p_receptor
#     (is R enriched in T?), using statistic mu[.,.] if the partner passes the expression filter, 0 otherwise.




def stack_binary(X):
    """it creates a side by side identical matrix where the first one will be used for the sum for the mean,
     the second half will be used for the fraction expressing."""
    Xb = X.copy()
    Xb.data[:] = 1.0
    return sp.hstack([X, Xb], format="csr")

def group_means(X, codes, n_groups, counts):
    """Mean of every gene of X inside every group of cells (celltype); dense (n_groups x ncols).
        it works in C language --> at the end it return a matrix containing K type of celltype with their mean expression
    """
    N = X.shape[0]
    H = sp.csr_matrix((np.ones(N, dtype=np.float32), (codes, np.arange(N))), shape=(n_groups, N))
    return (H @ X).toarray() / np.maximum(counts, 1)[:, None]

def means_and_fracs(Xcat, codes, n_groups, counts, G): #remember that Xcat is composed of 2 matrices where the first part is the real expression matrix for each gene and each cell, the second one is filled by 1 (and 0) where 1 is present if the cell express that gene
    """(mean, fraction expressing), both (G x K)."""

    mf = group_means(Xcat, codes, n_groups, counts)
    return mf[:, :G].T, mf[:, G:].T #the first one is the mean the second one is the fraction of the cells that express that gene in that cluster


def partner_matrix(gene_by_group, A, sub_idx):
    """from gene x group (cluster/celltype) matrix->to partner x group (cluster/celltype).
     gene -> protein through A, protein -> complex through MIN."""

    #remeber that sub_idx is a matrix of QxS where Q in the number of complex or protein (each row is composed of protein or complex) and
    # S is the number of columns, representing the MAXIMUM number of protein present in a single complex within the  dsatabase
    # quindi se nel database trovo che il complesso più grosso è formato da 10 proteine insieme, avro 10 colonne.
    #ogni cella sarà riempita con l'indice


    # =========================================================================================================
    # STRUTTURA DELLA MATRICE prot (Dimensione: (P + 1) x K = 1000 x K)
    # - P + 1 (Righe) : Proteine Reali (da 0 a P-1) + 1 Riga Fantasma di Padding all'indice P (es. 999)
    # - K (Colonne)   : Tipi Cellulari (es. T-cell, B-cell, Macrofagi, ecc.)
    #
    # Indice Riga (P) | Proteina / Identificatore | T-cell  | B-cell  | Macrofagi | Valori memorizzati
    # ---------------------------------------------------------------------------------------------------------
    # Indice 5        | Prot 3 del Trimero        | 2.10    | 0.50    | 1.80      | Espressione reale
    # Indice 6        | Prot 2 del Trimero        | 0.80    | 4.20    | 0.10      | Espressione reale
    # Indice 7        | Prot 1 del Trimero        | 5.40    | 1.10    | 3.20      | Espressione reale
    # Indice 12       | TGFB1 (Proteina Singola)  | 3.50    | 0.00    | 12.40     | Espressione reale
    # Indice 45       | CD45 - Subunità 1         | 8.90    | 7.10    | 0.00      | Espressione reale
    # Indice 88       | CD45 - Subunità 2         | 6.20    | 9.00    | 0.50      | Espressione reale
    # ...             | ...                       | ...     | ...     | ...       | ...
    # Indice 999 (P)  | Proteina Fantasma (Pad)   | +inf    | +inf    | +inf      | +inf (Padding fittizio)
    # =========================================================================================================



    # ===================================================================================================
    # STRUTTURA DELLA MATRICE sub_idx (Dimensione: Q x S = 3 x 3)
    # - Q (Righe)   : Numero di Partner molecolari (es. 3)
    # - S (Colonne) : Massimo numero di Subunità tra tutti i complessi (es. 3)
    # - P (Padding) : Indice dell'ultima riga della matrice prot che contiene +inf (es. 999)
    #
    # Partner (Q) | Tipo molecola              | Prot 1    | Prot 2    | Prot 3    | Riga in sub_idx (S=3)
    # ---------------------------------------------------------------------------------------------------
    # Partner 0   | Proteina Singola (TGFB1)   | Indice 12 | (Nessuna) | (Nessuna) | [12, 999, 999]
    # Partner 1   | Dimero (CD45)              | Indice 45 | Indice 88 | (Nessuna) | [45,  88, 999]
    # Partner 2   | Trimero (IL2 Recettore)    | Indice 5  | Indice 6  | Indice 7  | [ 5,   6,   7]
    # ===================================================================================================
    prot = A @ gene_by_group  # return (P x K) ---> becasue A (PxG) @ GxK --> PxK --> that is, protein expression for each cluster
    pad = np.full((1, prot.shape[1]), np.inf, dtype=prot.dtype) # fake protein used as padding with +inf value
    prot = np.vstack([prot, pad]) # index P == padding
    return prot[sub_idx].min(axis=1)  # (Q x S x K) -> min over S -> (Q x K)


def real_scores_and_tests(model, Xcat, codes, n_groups, expr_frac, min_score, autocrine):
    """Real scores and the list of testable (interaction, sender, receiver) triplets of one unit."""
    # ===================================================================================================
    # STRUTTURA DEL TENSORE / CUBO 3D IN real_scores_and_tests
    # Dimensioni del cubo: (N_interazioni x K_sender x K_receiver)
    #
    # Asse / Dimensione | Significato                        | Esempio di Indicizzazione
    # ---------------------------------------------------------------------------------------------------
    # Asse 0 (Righe)    | Indice Interazione (i)             | i = 105 (es. CXCL12 -> CXCR4)
    # Asse 1 (Piani)    | Tipo Cellulare Sender / Mittente (s)| s = 2   (es. Fibroblasti)
    # Asse 2 (Colonne)  | Tipo Cellulare Receiver / Ricevente| t = 5   (es. T-cell)
    #
    # Esempio di accesso: score[105, 2, 5] -> Punteggio dell'interazione CXCL12->CXCR4 da Fibroblasti a T-cell
    # ===================================================================================================
    counts = np.bincount(codes, minlength=n_groups)
    valid = counts > 0  # cell types present in this unit --> boolean array with lenght 'K' (clusters) with at least a cells for this specific unit (sample_ID, e.g. patient01_CRLM)
    means, fracs = means_and_fracs(Xcat, codes, n_groups, counts, len(model.genes))
    Pm = partner_matrix(means, model.A, model.sub_idx)   # matrix for the mean expression (Q x K)                 # (Q x K) mean expression
    Pf = partner_matrix(fracs, model.A, model.sub_idx)  # (Q x K) fraction expressing
    lig, rec = model.inter["lig"].to_numpy(), model.inter["rec"].to_numpy() #index for ligand and receptor--> so for each interaction it extracts the index of the partner <8protein or complexes) involeved in that interaction

    # (n_int x K x K) cubes: axis1 = sender, axis2 = receiver
    ok = (Pf[lig][:, :, None] >= expr_frac) & (Pf[rec][:, None, :] >= expr_frac) #boolean matrix  (N_intereaction x K_sender x K_recevier)

    score = 0.5 * (Pm[lig][:, :, None] + Pm[rec][:, None, :]) #mean of expression of the ligand in the sender e receptor in receiver for each cell in the matrix
    ok &= score > min_score #the interaction must be higher of a minumum score (if selected from the user)
    ok &= valid[None, :, None] & valid[None, None, :]
    if not autocrine: #Se l'utente non vuole analizzare la segnalazione autocrina (cellule che parlano con se stesse), usa la matrice identità np.eye per azzerare la diagonale dove sender==receiver.
        ok &= ~np.eye(n_groups, dtype=bool)[None, :, :]
    ii, ss, tt = np.nonzero(ok)        # ii = indici delle interazioni valide, ss= indici dei tipi cellulare sender , tt= indici dei tipi cellulare receiver                                   # coordinates of the tests
    return {"ii": ii, "ss": ss, "tt": tt, "lig_t": lig[ii], "rec_t": rec[ii],
            "real": score[ii, ss, tt].astype(np.float64),
            "real_lig": Pm[lig[ii], ss].astype(np.float64), "real_rec": Pm[rec[ii], tt].astype(np.float64),
            "frac_lig": Pf[lig[ii], ss], "frac_rec": Pf[rec[ii], tt]}






def permutation_counts(model, Xcat, codes, n_groups, tests, n_perm, seed, expr_frac, min_score):
    """Run the permutations
    . Returns an int array (3 x n_tests): for every test, how many nulls were
    >= the real statistic. Row 0: joint score; row 1: ligand in sender; row 2: receptor in receiver."""
    G = len(model.genes) #number of genes
    counts = np.bincount(codes, minlength=n_groups)
    rng = np.random.default_rng(seed)
    n_tests = tests["real"].shape[0]
    exceed = np.zeros((3, n_tests), dtype=np.int64) #matrix 3 x N_test --> 3 rows--> 1) how many 'fake' score are > than the real one, 2) how many times the expression of the ligand in the sender is > than the real one , 3) how many times the expression if the receptor in the receiver is > than the real one
    eps = 1e-9  # tolerance for floating-point ties
    lig_t, rec_t, ss, tt = tests["lig_t"], tests["rec_t"], tests["ss"], tests["tt"]

    for b in range(n_perm):
        perm = rng.permutation(codes)      #shulffle the labels in the sender and receiver.                               # shuffle the labels among the cells
        means, fracs = means_and_fracs(Xcat, perm, n_groups, counts, G)
        P = partner_matrix(means, model.A, model.sub_idx)
        F = partner_matrix(fracs, model.A, model.sub_idx)
        # same definition as in the real data: a partner that is not expressed enough counts as 0
        null_l = np.where(F[lig_t, ss] >= expr_frac, P[lig_t, ss], 0.0) #--> se nella permutazione corrente la frazione di cellule che esprime quel gene è minore a expr_frac allora il valore del partner (complesso/proteina) viene messo a 0 per ligand
        null_r = np.where(F[rec_t, tt] >= expr_frac, P[rec_t, tt], 0.0)#--> se nella permutazione corrente la frazione di cellule che esprime quel gene è minore a expr_frac allora il valore del partner (complesso/proteina) viene messo a 0 per receiver
        joint = np.where((null_l > 0) & (null_r > 0), 0.5 * (null_l + null_r), 0.0) # media aritmetica (0.5 x (L+R)
        joint = np.where(joint > min_score, joint, 0.0) #se uno dei due partner è 0 o se il punteggio non supera min_score (scelto da utente) allora la coppia viene azzerata
        exceed[0] += joint >= tests["real"] - eps #se il valore ottenuto per caso supera il valore real allora exceed incrementa di 1

        exceed[1] += null_l >= tests["real_lig"] - eps #stessa cosa per il ligando (quante volte l'espressione del ligando fake supera quella vera/osservata)
        exceed[2] += null_r >= tests["real_rec"] - eps #stessa cosa per il recettore
        if (b + 1) % 250 == 0: #ogni 250 permutazioni invia un messaggio nel log
            log.debug("  permutation %d/%d", b + 1, n_perm)
    return exceed

# DALLA MATRICE exceed AL P-VALUE DEFINITIVO
#
# Conteggio accumulato (exceed) | Formula p-value                 | Interpretazione biologica
# ---------------------------------------------------------------------------------------------------------
# exceed[0, j] (Score)         | p_score = exceed[0, j] / n_perm | Significatività complessiva dell'interazione
# exceed[1, j] (Ligando)       | p_lig   = exceed[1, j] / n_perm | Specificità d'espressione del Ligando (Sender)
# exceed[2, j] (Recettore)     | p_rec   = exceed[2, j] / n_perm | Specificità d'espressione del Recettore (Receiver)
# =========================================================================================================


def empirical_pvalues(exceed_counts, n_perm):
    """p = (1 + #{null >= real}) / (1 + n_perm)   (Phipson & Smyth 2010)."""
    return (1.0 + exceed_counts) / (1.0 + n_perm)


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]                    # monotone from the largest p downward
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out









# 4. INFERENCE DRIVER, REPLICATE CONSENSUS, NETWORK

KEY = ["condition", "interaction_id", "ligand", "receptor", "ligand_label", "receptor_label",
       "classification", "directionality", "sender", "receiver"]


def run_units(model, X, celltypes, units, *, min_cells, expr_frac, min_score, autocrine, n_perm, seed, correction):
    """it categorizes the celltypes with numerical codes (from 0 to K).
    for each 'unit' (sample_id) excludes those celltypes that do not have a 'min_cells' <nd extracts its expression matrix 'Xu' on which performs stack_binary --> real_score_and_tests --> permutation counts
     at the end computes the p-values.
     it returns 2 dataframes:
     1) Tested --> all the combinations tested with the associated results (sample, interaction, sender, receiver)
     2) presence --> valid celltypes in each sample, that will be used for the consensus step """
    # =========================================================================================================
    # STRUTTURA DEI DATAFRAME DI OUTPUT DI run_units
    #
    # 1. DataFrame tested (Risultati dei test statistici):
    # Colonna        | Tipo   | Significato
    # ---------------------------------------------------------------------------------------------------------
    # condition      | str    | Condizione sperimentale (es. Controllo vs Trattato)
    # sample         | str    | Identificativo del singolo campione / replicato
    # interaction_id | str    | ID dell'interazione nel database (es. CPI-SS001A)
    # sender         | str    | Nome del tipo cellulare mittente (es. Fibroblast)
    # receiver       | str    | Nome del tipo cellulare ricevente (es. T-cell)
    # score          | float  | Punteggio medio d'espressione reale (Pm[L,S] + Pm[R,T]) / 2
    # pval           | float  | P-value empirico grezzo dal test di permutazione
    # padj           | float  | P-value corretto per FDR (Benjamini-Hochberg)
    #
    # 2. DataFrame presence (Registro di usabilità cellulare):
    # Colonna        | Tipo   | Significato
    # ---------------------------------------------------------------------------------------------------------
    # condition      | str    | Condizione sperimentale
    # sample         | str    | Identificativo del campione
    # celltype       | str    | Tipo cellulare che ha superato la soglia min_cells nel campione
    # =========================================================================================================
    cats = pd.Categorical(celltypes.astype(str))  # ONE global list of cell types: column k = same type everywhere
    all_types = list(cats.categories)
    codes_all = np.asarray(cats.codes)
    K = len(all_types) #total number of celltypes
    log.info("%d cell types: %s", K, all_types)

    out, presence = [], []
    for u_i, u in enumerate(units): #for each unit
        t0 = time.time()
        codes = codes_all[u.cell_idx]
        sizes = np.bincount(codes, minlength=K) #how many cells in a specific celltype in sample
        keep_type = sizes >= min_cells  #drop cell types with too few cells IN THIS unit
        if keep_type.sum() < 2: #if there are less than 2 different celltypes --> it will be skipped (no communication)
            log.warning("[%s | %s] fewer than 2 cell types with >= %d cells: unit skipped.",
                        u.condition, u.sample, min_cells)
            continue
        keep_cell = keep_type[codes]
        Xu, cu = X[u.cell_idx[keep_cell]], codes[keep_cell]
        presence += [(u.condition, u.sample, all_types[k]) for k in np.nonzero(keep_type)[0]] # registers which celltypes are usable.

        Xcat = stack_binary(Xu)  # [expression | expressed(0/1) for each unit]
        tests = real_scores_and_tests(model, Xcat, cu, K, expr_frac, min_score, autocrine) #building the tensor 3D
        n_tests = tests["real"].size
        if n_tests == 0:
            log.warning("[%s | %s] no testable interaction.", u.condition, u.sample)
            continue

        #permutation test and p-value adjusted
        exceed = permutation_counts(model, Xcat, cu, K, tests, n_perm, seed + u_i, expr_frac, min_score)
        p = empirical_pvalues(exceed[0], n_perm)# computing the p-value (empirical)
        p_lig = empirical_pvalues(exceed[1], n_perm)# ligand enrichment in the sender
        p_rec = empirical_pvalues(exceed[2], n_perm) # receptor enrichment in the receiver
        padj = bh_adjust(p) if correction == "fdr_bh" else p

        inter = model.inter.iloc[tests["ii"]]
        types = np.asarray(all_types, dtype=object)

        #concatening all the dataframe
        out.append(pd.DataFrame({
            "condition": u.condition, "sample": u.sample,
            "interaction_id": inter["interaction_id"].to_numpy(),
            "ligand": inter["ligand"].to_numpy(), "receptor": inter["receptor"].to_numpy(),
            "ligand_label": inter["ligand_label"].to_numpy(), "receptor_label": inter["receptor_label"].to_numpy(),
            "classification": inter["classification"].to_numpy(), "directionality": inter["directionality"].to_numpy(),
            "sender": types[tests["ss"]], "receiver": types[tests["tt"]],
            "score": tests["real"], "frac_ligand": tests["frac_lig"], "frac_receptor": tests["frac_rec"],
            "pval": p, "padj": padj, "p_ligand": p_lig, "p_receptor": p_rec}))
        log.info("[%d/%d] %s | %s: %d cells, %d tests, %.1f s", u_i + 1, len(units), u.condition, u.sample,
                 len(cu), n_tests, time.time() - t0)

    if not out:
        raise RuntimeError("No unit produced results: relax --min-cells / --expr-frac.")
    presence = pd.DataFrame(presence, columns=["condition", "sample", "celltype"])
    return pd.concat(out, ignore_index=True), presence






def consensus(tested, presence, *, alpha, min_samples, min_sample_frac, require_specific=True):
    """Aggregate the samples of each condition and flag the CONFIDENT interactions.

    A triplet (interaction, S, T) is significant in a sample if padj <= alpha (and, with require_specific, also
    p_ligand <= alpha and p_receptor <= alpha).
    an interaction between 2 celltypes is considered CONFIDENT in a condition if it is significant in at least max(min_samples, ceil(min_sample_frac * n_eligible), 1),
    --> that is: it is considered confident if the number of that specific interaction between 2 celltypes is significant (p-value<alpha) in at least:
    - min_sample_frac =0.5 --> 50% of samples eligible (where sample eligibles are the number of samples where the 2 celltypes have at least min_cells --> ad esempio in 4 pazienti testati solo 3 hanno che nel CRC i macrofagi e T cells superano un numero minimo di cellule nel cluster)--> 3 samples eligibles
    (samples of that condition in which BOTH cell types had >= min_cells cells)
    - or min_sample (decided a priori)
    - 1 sample
    """
    # =========================================================================================================
    # STRUTTURA METRICHE DI CONSENSO NEL DATAFRAME AGGREGATO (agg)
    #
    # Colonna            | Tipo   | Significato Logico e Calcolo
    # ---------------------------------------------------------------------------------------------------------
    # n_samples_tested   | int    | Numero di campioni in cui l'interazione e' stata effettivamente testata
    # n_samples_eligible | int    | Numero di campioni in cui SIA Sender CHE Receiver superavano min_cells
    # n_samples_sig      | int    | Numero di campioni in cui l'interazione e' risultata significativa
    # need               | float  | Soglia minima di campioni significativi richiesti per la condizione
    # confident          | bool   | Flag finale: True se n_samples_sig >= need (Interazione valida per la condizione)
    # =========================================================================================================
    t = tested.copy()
    t["significant"] = t["padj"] <= alpha
    if require_specific: #if requested by the user
        t["significant"] &= (t["p_ligand"] <= alpha) & (t["p_receptor"] <= alpha)

    agg = (t.groupby(KEY, observed=True, sort=False, dropna=False)
           .agg(n_samples_tested=("sample", "nunique"), n_samples_sig=("significant", "sum"),
                mean_score=("score", "mean"), best_padj=("padj", "min"), best_pval=("pval", "min"))
           .reset_index())

    where = presence.groupby(["condition", "celltype"])["sample"].apply(set).to_dict()
    pairs = agg[["condition", "sender", "receiver"]].drop_duplicates()
    elig = {(c, s, r): len(where.get((c, s), set()) & where.get((c, r), set()))
            for c, s, r in pairs.itertuples(index=False)}
    agg["n_samples_eligible"] = [elig[(c, s, r)] for c, s, r in zip(agg["condition"], agg["sender"], agg["receiver"])]
    need = np.maximum(np.maximum(min_samples, np.ceil(min_sample_frac * agg["n_samples_eligible"])), 1)
    agg["confident"] = agg["n_samples_sig"] >= need
    return agg






def build_network(conf):
    """From the confident interactions (confident ==True) to a network (edges, node metrics).
    It answers the question: 'how are structured the communications between different cell-types within the tissue?'


    -1)edge (condition, sender -> receiver):
        -a) n_interactions = number of distinct confident L-R pairs (edge WIDTH),
        -b) strength = sum of their mean scores.
    -2)node: out/in degree (with how many DIFFERENT cell types it talks), out/in strength (total confident pairs
          sent/received). Autocrine loops are reported separately."""
    c = conf[conf["confident"]] #confident interactions
    if c.empty:
        return pd.DataFrame(), pd.DataFrame()

    #for each triplet (condition, sender and receiver) it computes:
    # 1) n_interaction--> it will be the WIDTH of the edge
    #2) strength= sum of mean_score of ALL the interactions between those 2 celltypes in a condition
    edges = (c.groupby(["condition", "sender", "receiver"], observed=True).agg(n_interactions=("interaction_id", "size"), strength=("mean_score", "sum")).reset_index())
    # 1. DataFrame EDGES (Archi del Grafo):
    # Colonna        | Tipo   | Significato
    # ---------------------------------------------------------------------------------------------------------
    # condition      | str    | Condizione sperimentale (es. Controllo vs Trattato)
    # sender         | str    | Nodo Sorgente / Cellula Mittente
    # receiver       | str    | Nodo Destinazione / Cellula Ricevente
    # n_interactions | int    | Numero di coppie L-R confidenti attive su questo arco (Spessore Arco)
    # strength       | float  | Somma dei mean_score d'espressione per tutte le L-R su questo arco (Peso Arco)

    rows = []

    #for each cell type in a specific condition the function breaks down the communication in 3 different categories:
    #1) Outbound Paracrine Signaling (paracrina in uscita)
    #2) incoming paracrine signaling
    #3) autocrin
    for cond, e in edges.groupby("condition"):
        for ct in sorted(set(e["sender"]) | set(e["receiver"])):
            out_e = e[(e["sender"] == ct) & (e["receiver"] != ct)] #different celltypes
            in_e = e[(e["receiver"] == ct) & (e["sender"] != ct)] #differnt celltypes
            auto = e[(e["sender"] == ct) & (e["receiver"] == ct)] #the same celltype

            #from those 3 categories it extracts 5 metrics:
            # ---------------------------------------------------------------------------------------------------------
            # celltype           | str    | Tipo cellulare analizzato come nodo del grafo

            # out_degree         | int    | Numero di tipi cellulari DISTINTI a cui invia segnali
            # out_strength       | int    | Conteggio totale di interazioni L-R inviate verso l'esterno

            # in_degree          | int    | Numero di tipi cellulari DISTINTI da cui riceve segnali
            # in_strength        | int    | Conteggio totale di interazioni L-R ricevute dall'esterno

            # autocrine_strength | int    | Conteggio di interazioni L-R autocrine (cellula parla con se stessa)
            # =========================================================================================================
            # it is usefull for the node's centrality
            rows.append({"condition": cond, "celltype": ct, "out_degree": len(out_e), "in_degree": len(in_e),
                         "out_strength": int(out_e["n_interactions"].sum()),
                         "in_strength": int(in_e["n_interactions"].sum()),
                         "autocrine_strength": int(auto["n_interactions"].sum())})
    return edges, pd.DataFrame(rows)










# 5. PLOTS

# 1 network : circular directed graph, arrow = sender -> receiver, width = # confident L-R pairs
# 2 heatmap : sender x receiver matrix of the # of confident pairs
# 3 dotplot : WHICH ligand-receptor pairs stand behind the strongest edges (size = -log10 p, colour = score)
# 4 cascade : "starting from cell type X, who does it talk to, and who do THOSE talk to?" (BFS layers)

def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def _palette(all_types):
    """One fixed color per cell type, identical in all plots and conditions."""
    cmap = plt.get_cmap("tab20")
    return {ct: cmap(i % 20) for i, ct in enumerate(sorted(all_types))}


def _select_edges(e, min_edge, top_edges):
    return e[e["n_interactions"] >= min_edge].sort_values("n_interactions", ascending=False).head(top_edges)


def _save(fig, outdir, stem, formats):
    for f in formats:
        path = os.path.join(outdir, f"{stem}.{f}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        log.info("saved %s", path)
    plt.close(fig)


def plot_network(edges, colors, title, min_edge=1, top_edges=60):
    """it return a circular plot with oriented communication.
    1) the size of the node (celltype) is proportional to the total number of signals (sent and received).
    2) edges : from sender to receiver
    3)width of the edge (lw) -> proportional to number of valid interaction (n_interaction)
    4) autocrin loop
    """
    e = _select_edges(edges, min_edge, top_edges)
    if e.empty:
        return None
    nodes = sorted(set(e["sender"]) | set(e["receiver"]))
    n = len(nodes)
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n # start at 12 o'clock, clockwise
    pos = {ct: np.array([np.cos(a), np.sin(a)]) for ct, a in zip(nodes, ang)}
    tot = {ct: 0 for ct in nodes} # node size ~ pairs sent + received
    for _, r in e.iterrows():
        tot[r["sender"]] += r["n_interactions"]
        if r["receiver"] != r["sender"]:
            tot[r["receiver"]] += r["n_interactions"]
    tmax = max(tot.values())
    size = {ct: 250 + 1500 * tot[ct] / tmax for ct in nodes}
    rad_pt = {ct: np.sqrt(size[ct]) / 2 for ct in nodes}  # radius in points, to shrink arrows
    fig, ax = plt.subplots(figsize=(9, 9))
    wmax = e["n_interactions"].max()
    for _, r in e.iterrows():
        s, t, w = r["sender"], r["receiver"], r["n_interactions"]
        lw = 0.6 + 5.5 * w / wmax
        if s == t: # autocrine: small loop outside the node
            ax.add_patch(Circle(pos[s] * 1.13, 0.07, fill=False, ec=colors[s], lw=lw, alpha=0.7))
            continue
        ax.add_patch(FancyArrowPatch(pos[s], pos[t], connectionstyle="arc3,rad=0.25", arrowstyle="-|>",
                                     mutation_scale=9 + 1.5 * lw, lw=lw, color=colors[s], alpha=0.6,
                                     shrinkA=rad_pt[s] + 1, shrinkB=rad_pt[t] + 3, zorder=1))
    for ct in nodes:
        ax.scatter(*pos[ct], s=size[ct], color=colors[ct], edgecolor="black", linewidth=0.8, zorder=3)
        x, y = pos[ct] * 1.30
        ax.text(x, y, ct, ha="left" if x > 0.15 else "right" if x < -0.15 else "center", va="center",
                fontsize=10, zorder=4)
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13)
    vals = sorted({max(1, int(wmax) // 4), max(1, int(wmax) // 2), int(wmax)})
    ax.legend(handles=[Line2D([0], [0], color="grey", lw=0.6 + 5.5 * v / wmax, label=f"{v} pairs") for v in vals],
              loc="lower left", frameon=False, title="edge width")
    return fig


def plot_heatmap(edges, title):
    """NxN matrix (for each celltype in a specific condition) that quantifies the density communication"""
    if edges.empty:
        return None
    types = sorted(set(edges["sender"]) | set(edges["receiver"]))
    idx = {ct: i for i, ct in enumerate(types)}
    M = np.zeros((len(types), len(types)))
    for _, r in edges.iterrows():
        M[idx[r["sender"]], idx[r["receiver"]]] = r["n_interactions"]
    order = np.argsort(-(M.sum(0) + M.sum(1))) # busiest cell types first
    M = M[np.ix_(order, order)]
    labels = [types[i] for i in order]
    size = max(5.0, 0.45 * len(types) + 2.5)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(M, cmap="magma_r")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")
    ax.set_title(title)
    if len(labels) <= 25:
        for i in range(len(labels)):
            for j in range(len(labels)):
                if M[i, j] > 0:
                    ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] > M.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="# confident L-R pairs")
    return fig


def plot_dotplot(conf, title, top_n=30, max_pairs=20):
    """it shows which molecules explain the strongest connection between the celltypes with mean interaction score (color) and
    size --> where the size is defined from the p-value  --> the smaller the p-value of the permutation the bigger the size of the dot.
    """
    c = conf[conf["confident"]].copy()
    if c.empty:
        return None
    c["pair"] = c["sender"] + " \u2192 " + c["receiver"]
    c["lr"] = c["ligand_label"].astype(str) + " \u2192 " + c["receptor_label"].astype(str)
    pairs = c.groupby("pair").size().nlargest(max_pairs).index.tolist() # busiest cell-type pairs
    c = c[c["pair"].isin(pairs)]
    lr_rank = (c.groupby("lr").agg(n=("pair", "size"), s=("mean_score", "max"))
               .sort_values(["n", "s"], ascending=False).head(top_n).index.tolist())
    c = c[c["lr"].isin(lr_rank)]
    xi = {p: i for i, p in enumerate(pairs)}
    yi = {l: i for i, l in enumerate(lr_rank[::-1])}                       # best on top
    nlp = -np.log10(np.clip(c["best_pval"].to_numpy(dtype=float), 1e-6, 1))
    fig, ax = plt.subplots(figsize=(0.55 * len(pairs) + 5, 0.3 * len(lr_rank) + 3))
    sc = ax.scatter(c["pair"].map(xi), c["lr"].map(yi), s=25 + 70 * nlp, c=c["mean_score"], cmap="viridis",
                    edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(pairs)), pairs, rotation=60, ha="right")
    ax.set_yticks(range(len(lr_rank)), lr_rank[::-1])
    ax.set_xlim(-0.7, len(pairs) - 0.3)
    ax.set_ylim(-0.7, len(lr_rank) - 0.3)
    ax.grid(alpha=0.25)
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="mean interaction score")
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc="lightgrey", mec="black", ms=np.sqrt(25 + 70 * v) * 0.9, label=f"{v}") for v in (1, 2, 3)], title="-log10(best p)", loc="upper left", bbox_to_anchor=(1.25, 1.0), frameon=False)
    return fig


def plot_cascade(edges, source, colors, title, max_hops=3, min_edge=3):

    """it shows a stratified diagram that tracks the signal propagation starting from a source (celltype).
    it has 3 levels:
    1) source --> starting celltype
    2) hop 1 --> the algorithm looks for all the celltype that directly receive signal from the source (for example if the CD8 is the source, it looks for all the celltypes that act as receiver from the CD8)
    3) hop 2 --> the same as before but starting from hop_1 not reached before
    the size of the edges is proportional to n_interactions (how many valid LR pairs connect the 2 celltype)
    """
    e = edges[(edges["n_interactions"] >= min_edge) & (edges["sender"] != edges["receiver"])]
    if source not in set(e["sender"]) | set(e["receiver"]):
        log.warning("cascade: '%s' has no edge with >= %d pairs in this condition.", source, min_edge)
        return None
    adj = {}
    for _, r in e.iterrows():
        adj.setdefault(r["sender"], []).append(r["receiver"])
    dist, q = {source: 0}, deque([source])
    while q:
        u = q.popleft()
        if dist[u] >= max_hops:
            continue
        for v in adj.get(u, []):
            if v not in dist: ## <--- Vengono considerate SOLO le cellule (cluster) NON ancora raggiunte prima
                dist[v] = dist[u] + 1
                q.append(v)
    layers = {}
    for ct, h in dist.items():
        layers.setdefault(h, []).append(ct)
    fwd = e[e.apply(lambda r: r["sender"] in dist and r["receiver"] in dist
                              and dist[r["receiver"]] == dist[r["sender"]] + 1, axis=1)]
    inw = fwd.groupby("receiver")["n_interactions"].sum().to_dict()
    for h in layers:
        layers[h].sort(key=lambda ct: -inw.get(ct, 0))
    H, Lmax, dx = max(layers), max(len(v) for v in layers.values()), 3.4
    fig, ax = plt.subplots(figsize=(dx * H + 3, 1.0 * Lmax + 2))
    fig.subplots_adjust(0, 0, 1, 0.93)
    pos = {ct: (h * dx, -(i - (len(nodes) - 1) / 2)) for h, nodes in layers.items() for i, ct in enumerate(nodes)}
    wmax = fwd["n_interactions"].max() if len(fwd) else 1
    for _, r in fwd.iterrows():
        (x1, y1), (x2, y2) = pos[r["sender"]], pos[r["receiver"]]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6 + 5.0 * r["n_interactions"] / wmax,
                                    color=colors[r["sender"]], alpha=0.65, shrinkA=34, shrinkB=34,
                                    connectionstyle="arc3,rad=0.08"))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, int(r["n_interactions"]), fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    for ct, (x, y) in pos.items():
        ax.text(x, y, ct, ha="center", va="center", fontsize=10, fontweight="bold" if ct == source else None,
                bbox=dict(boxstyle="round,pad=0.4", fc=colors.get(ct, "lightgrey"), ec="black", alpha=0.75))
    for h in layers:
        ax.text(h * dx, Lmax / 2 + 0.6, "source" if h == 0 else f"hop {h}", ha="center", fontsize=11, color="grey")
    ax.set_xlim(-1.3, H * dx + 1.3)
    ax.set_ylim(-Lmax / 2 - 0.8, Lmax / 2 + 1.1)
    ax.axis("off")
    ax.set_title(title, fontsize=13)
    return fig


def make_all_plots(edges, conf, outdir, formats=("png",), min_edge=3, top_edges=60, cascade_sources=(),
                   max_hops=3, dot_top=30):
    """Every plot, for every condition."""
    if edges.empty:
        log.warning("No confident interaction: no plot produced.")
        return
    os.makedirs(outdir, exist_ok=True)
    colors = _palette(sorted(set(edges["sender"]) | set(edges["receiver"])))
    for cond, e in edges.groupby("condition"):
        tag, c = _safe(cond), conf[conf["condition"] == cond]
        figs = {f"network_{tag}": plot_network(e, colors, f"Cell-cell communication: {cond}", min_edge, top_edges),
                f"heatmap_{tag}": plot_heatmap(e, f"Confident L-R pairs: {cond}"),
                f"dotplot_{tag}": plot_dotplot(c, f"Top ligand-receptor pairs: {cond}", top_n=dot_top)}
        for src in cascade_sources:
            figs[f"cascade_{tag}_from_{_safe(src)}"] = plot_cascade(
                e, src, colors, f"Signalling cascade from {src}: {cond}", max_hops, min_edge)
        for stem, fig in figs.items():
            if fig is not None:
                _save(fig, outdir, stem, formats)



# 6. FAKE DATA GENERATOR  (python celltalkpy.py fake-data ...)

# Realistic fake AnnData, same column names as a typical CRC / CRLM study:
# obs['celltype'] 9 types (PBMC has immune cells only) | obs['Patient'] P1..P3, each with ALL tissues |
# obs['Tumor_type_recurrence'] CRC / CRLM / PBMC -->sample = Patient + Tumor_type_recurrence
#   X = 1e4-normalised + log1p ; layers['counts'] = raw counts ; var_names = HGNC symbols.
# PLANTED signals are written to <out>_ground_truth.csv; everything else is noise that celltalkpy should NOT report.

PLANTED = [  # (ligand, receptor, sender celltype, receiver celltype, tissues where the signal exists)
    ("CXCL9", "CXCR3", "Macrophage", "CD8 T", ["CRC", "CRLM"]),
    ("CD274", "PDCD1", "Macrophage", "CD8 T", ["CRC", "CRLM"]),
    ("CSF1", "CSF1R", "Tumor epithelial", "Macrophage", ["CRC", "CRLM"]),
    ("VEGFA", "KDR", "Tumor epithelial", "Endothelial", ["CRC", "CRLM"]),
    ("CXCL12", "CXCR4", "Fibroblast", "CD8 T", ["CRC", "CRLM"]),
    ("CCL2", "CCR2", "Fibroblast", "Macrophage", ["CRLM"]),# CRLM-specific
    ("CXCL16", "CXCR6", "Macrophage", "gd T", ["CRC"]), # CRC-specific
    ("CCL5", "CCR5", "CD8 T", "Macrophage", ["CRC", "CRLM", "PBMC"]),
]
MARKERS = {"CD8 T": ["CD3E", "CD8A", "CD8B"], "gd T": ["CD3E", "TRDC", "TRGC1"], "CD4 T": ["CD3E", "CD4", "IL7R"],
           "NK": ["NKG7", "KLRD1", "GNLY"], "B cell": ["MS4A1", "CD79A"], "Macrophage": ["CD68", "C1QA", "C1QB"],
           "Tumor epithelial": ["EPCAM", "KRT8", "KRT18"], "Fibroblast": ["COL1A1", "DCN", "LUM"],
           "Endothelial": ["PECAM1", "VWF", "CLDN5"]}
CELLTYPES = list(MARKERS)
IMMUNE = ["CD8 T", "gd T", "CD4 T", "NK", "B cell", "Macrophage"]
TISSUES = {"CRC": CELLTYPES, "CRLM": CELLTYPES, "PBMC": IMMUNE}
PATIENTS = ["P1", "P2", "P3"]


def make_fake_adata(cpdb_dir, out, cells_per_type=90, n_noise_genes=1500, seed=0):
    rng = np.random.default_rng(seed)
    db_genes = _read_table(cpdb_dir, "gene_input.csv")["hgnc_symbol"].dropna().unique().tolist()
    extra = [g for lst in MARKERS.values() for g in lst] + [g for p in PLANTED for g in p[:2]]
    genes = list(dict.fromkeys(db_genes + extra + [f"NOISE{i:04d}" for i in range(n_noise_genes)]))
    gi, G = {g: i for i, g in enumerate(genes)}, len(genes)

    base = np.exp(rng.normal(-2.0, 1.6, G))# ~0.01 ... ~10 counts per cell
    for g in extra:  # markers / planted genes: silent unless switched on
        base[gi[g]] = 0.03

    def program(ct, tissue): # multiplicative rates of a cell type in a tissue
        v = np.ones(G)
        for m in MARKERS[ct]:
            v[gi[m]] = 100.0
        for lig, rec, s, r, tissues in PLANTED:
            if tissue in tissues:
                if ct == s:
                    v[gi[lig]] = 100.0
                if ct == r:
                    v[gi[rec]] = 100.0
        return v

    blocks, obs_rows = [], []
    for pat in PATIENTS:
        for tissue, types in TISSUES.items():
            sample_effect = np.exp(rng.normal(0, 0.15, G)) # patient/tissue variability
            for ct in types:
                n = max(30, int(rng.normal(cells_per_type, cells_per_type * 0.25)))
                size = np.exp(rng.normal(0, 0.25, n)) # per-cell sequencing depth
                counts = rng.poisson(size[:, None] * (base * program(ct, tissue) * sample_effect)[None, :])
                blocks.append(sp.csr_matrix(counts.astype(np.float32)))
                obs_rows += [(ct, pat, tissue)] * n
    counts = sp.vstack(blocks, format="csr")
    obs = pd.DataFrame(obs_rows, columns=["celltype", "Patient", "Tumor_type_recurrence"])
    obs.index = [f"cell{i:06d}" for i in range(len(obs))]
    for c in obs.columns:
        obs[c] = obs[c].astype("category")

    tot = np.asarray(counts.sum(axis=1)).ravel()  # standard normalisation: 1e4 per cell + log1p
    tot[tot == 0] = 1
    norm = sp.diags(1e4 / tot) @ counts
    norm.data = np.log1p(norm.data)
    adata = ad.AnnData(X=norm.astype(np.float32).tocsr(), obs=obs, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = counts
    adata.write_h5ad(out, compression="gzip")

    truth = pd.DataFrame([(l, r, s, t, c) for l, r, s, t, cs in PLANTED for c in cs],
                         columns=["ligand", "receptor", "sender", "receiver", "condition"])
    truth_path = os.path.splitext(out)[0] + "_ground_truth.csv"
    truth.to_csv(truth_path, index=False)
    log.info("%d cells x %d genes -> %s ; planted signals -> %s", adata.n_obs, adata.n_vars, out, truth_path)



# 7. COMMAND LINE INTERFACE

def _add_plot_args(p):
    g = p.add_argument_group("plots")
    g.add_argument("--plot-format", nargs="+", default=["png"], choices=["png", "pdf", "svg"],
                   help="Output format(s) of the figures.")
    g.add_argument("--plot-min-edge", type=int, default=3,
                   help="Draw only edges with at least this many confident L-R pairs.")
    g.add_argument("--plot-top-edges", type=int, default=60, help="Draw at most this many edges in the network plot.")
    g.add_argument("--plot-top-lr", type=int, default=30, help="Number of ligand-receptor pairs (rows) in the dotplot.")
    g.add_argument("--cascade-source", nargs="+", default=[],
                   help="Cell type(s) to start the cascade plot from, e.g. --cascade-source 'CD8 T' Tumor.")
    g.add_argument("--max-hops", type=int, default=3, help="Maximum number of steps of the cascade plot.")


def build_parser():
    p = argparse.ArgumentParser(
        prog="python celltalkpy.py",
        description="Cell-cell communication inference from the CellPhoneDB database.")
    p.add_argument("--version", action="version", version=f"celltalkpy {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="Print debug messages.")
    sub = p.add_subparsers(dest="command", required=True)
    fmt = argparse.ArgumentDefaultsHelpFormatter

#### running part:
    r = sub.add_parser("run", help="Run the full analysis.", formatter_class=fmt)
    g = r.add_argument_group("input")

    g.add_argument("--adata", required=True, help="Path to the .h5ad file (normalised, log1p expression).")

    g.add_argument("--cpdb-dir", required=True, help="Folder with gene_input.csv, protein_input.csv, complex_input.csv, interaction_input.csv (for cellphonedb-data-5.0.0: its data/ sub-folder), or with a cellphonedb.zip containing them.")

    g.add_argument("--celltype-col", required=True, help="adata.obs column with the cell types.")

    g.add_argument("--sample-col", nargs="+", default=None,help="OPTIONAL adata.obs column(s) defining a biological sample (replicate). Several columns are concatenated, e.g. --sample-col Patient Tumor_type. Each sample is tested separately. If comitted, all cells of a condition are pooled")

    g.add_argument("--condition-col", default=None,help="OPTIONAL adata.obs column with the condition (e.g. CRC / CRLM). Results are aggregated and plotted per condition.")

    g.add_argument("--layer", default=None, help="Use this adata.layers entry instead of adata.X.")

    g.add_argument("--use-raw", action="store_true", help="Use adata.raw.X (must be normalised/log1p).")

    g.add_argument("--gene-col", default="index", help="'index' = adata.var_names, otherwise the adata.var column holding gene names/IDs.")

    g.add_argument("--gene-id-type", choices=["symbol", "ensembl"], default="symbol",help="Type of the gene identifiers in the data.")



#### stastics part:
    g = r.add_argument_group("statistics")
    g.add_argument("--min-cells", type=int, default=10, help="Ignore cell types with fewer cells than this (within each sample/unit).")

    g.add_argument("--expr-frac", type=float, default=0.1,help="A ligand/receptor is 'expressed' in a cell type if >= this fraction of its cells express it (all subunits, for complexes)")

    g.add_argument("--min-score", type=float, default=0.0,
                   help="Minimum interaction score (mean expression of ligand and receptor) to be tested.")

    g.add_argument("--no-autocrine", action="store_true", help="Do not test a cell type communicating with itself.")

    g.add_argument("--n-perm", type=int, default=1000, help="Number of permutations. The smallest possible p-value is 1/(n_perm+1)")

    g.add_argument("--correction", choices=["none", "fdr_bh"], default="none", help="Multiple-testing correction of the joint p-value within each sample. WARNING: fdr_bh over ~10^4-10^5 tests needs n_perm >> 1000, otherwise the smallest possible p-value can never pass the FDR and NOTHING is significant.")

    g.add_argument("--alpha", type=float, default=0.05, help="Significance threshold on the (adjusted) p-values.")

    g.add_argument("--require-specific", action=argparse.BooleanOptionalAction, default=True, help="an interaction is significant only if, besides the joint test, the ligand is enriched in the sender AND the receptor in the receiver (p <= alpha).\n--no-require-specific gives the CellPhoneDB-like behaviour.")

    g.add_argument("--min-samples", type=int, default=1,
                   help="An interaction is CONFIDENT in a condition if significant in at least this many samples... (default: 1)")

    g.add_argument("--min-sample-frac", type=float, default=1.0, help="An interaction is CONFIDENT in a condition if significant in at least this fraction (0-1) of the samples where both cell types exist-->(1.0 = all of them; lower it, e.g. 0.6, when you have many samples).")

    g.add_argument("--seed", type=int, default=0, help="Random seed (same seed -> same results) (due to permutation randomization)")

    g.add_argument("--negative-control", action="store_true", help="it is a sanity check: shuffle the cell-type labels of the whole dataset BEFORE the analysis. Real biology is destroyed, so you should have ~0 confident interactions\nif not, thresholds are too permissive for your data.")
#### output part:
    g = r.add_argument_group("output")
    g.add_argument("--outdir", default="celltalkpy_out", help="Output folder (created if missing).")
    g.add_argument("--no-plots", action="store_true", help="Only write the tables.")
    _add_plot_args(r)
    r.set_defaults(func=cmd_run)

#### plot part:
    q = sub.add_parser("plot", help="Redraw the plots of a previous run (no recomputation).", formatter_class=fmt)
    q.add_argument("--outdir", required=True, help="Folder of a previous `run`.")
    _add_plot_args(q)
    q.set_defaults(func=cmd_plot)

#### fake-data generation part
    f = sub.add_parser("fake-data", help="Generate a fake AnnData with planted signals to try the script.", formatter_class=fmt)
    f.add_argument("--cpdb-dir", required=True, help="Same folder as for `run` (used to take real gene symbols).")
    f.add_argument("--out", default="fake_adata.h5ad", help="Output .h5ad (a *_ground_truth.csv is written beside it).")
    f.add_argument("--cells-per-type", type=int, default=90, help="Mean number of cells per cell type per sample.")
    f.add_argument("--n-noise-genes", type=int, default=1500, help="Extra genes not in the database.")
    f.add_argument("--seed", type=int, default=0, help="Random seed.")
    f.set_defaults(func=cmd_fake)
    return p


def cmd_run(a):
    t0 = time.time()
    os.makedirs(a.outdir, exist_ok=True)

    # 1) expression + database
    log.info("Reading %s", a.adata)
    adata = ad.read_h5ad(a.adata)
    if a.celltype_col not in adata.obs.columns:
        sys.exit(f"--celltype-col '{a.celltype_col}' not in adata.obs. Columns: {list(adata.obs.columns)}")
    X, gene_names = load_expression(adata, a.layer, a.use_raw, a.gene_col)
    if a.negative_control:
        rng = np.random.default_rng(a.seed)
        adata.obs[a.celltype_col] = rng.permutation(adata.obs[a.celltype_col].astype(str).to_numpy())
        log.warning("NEGATIVE CONTROL: cell-type labels have been shuffled. Results are meaningless by design.")
    tables = load_cpdb(a.cpdb_dir)
    model = build_model(tables, gene_names, a.gene_id_type)
    col_of = {g: i for i, g in enumerate(gene_names)}
    X = X[:, [col_of[g] for g in model.genes]].tocsr()# keep only database genes

    # 2) samples / conditions  = make_units(adata.obs, adata.sample_col, adata.condition_col)
    units = make_units(adata.obs, a.sample_col, a.condition_col)
    # 3) statistics
    tested, presence = run_units(model, X, adata.obs[a.celltype_col], units, min_cells=a.min_cells,
                                 expr_frac=a.expr_frac, min_score=a.min_score, autocrine=not a.no_autocrine,
                                 n_perm=a.n_perm, seed=a.seed, correction=a.correction)
    conf = consensus(tested, presence, alpha=a.alpha, min_samples=a.min_samples,
                     min_sample_frac=a.min_sample_frac, require_specific=a.require_specific)
    edges, nodes = build_network(conf)

    # 4) tables
    o = a.outdir
    tested.to_csv(os.path.join(o, "tested_interactions.csv.gz"), index=False)
    conf.to_csv(os.path.join(o, "consensus_interactions.csv.gz"), index=False)
    conf[conf["confident"]].to_csv(os.path.join(o, "confident_interactions.csv"), index=False)# the main result
    edges.to_csv(os.path.join(o, "network_edges.csv"), index=False)
    nodes.to_csv(os.path.join(o, "network_node_metrics.csv"), index=False)
    with open(os.path.join(o, "run_parameters.json"), "w") as fh:
        json.dump({**{k: v for k, v in vars(a).items() if k != "func"}, "version": __version__}, fh, indent=2)
    n_conf = int(conf["confident"].sum())
    log.info("%d confident (interaction, sender, receiver, condition) triplets; %d network edges.", n_conf, len(edges))
    if n_conf == 0 and not a.negative_control:
        log.warning("Nothing is confident. Try: lower --min-sample-frac (e.g. 0.6) or --min-samples 1, lower --expr-frac, or --no-require-specific; check that X is log-normalised.")

    # 5) plots
    if not a.no_plots:
        make_all_plots(edges, conf, os.path.join(o, "plots"), a.plot_format, a.plot_min_edge, a.plot_top_edges,
                       a.cascade_source, a.max_hops, a.plot_top_lr)
    log.info("Done in %.1f min.", (time.time() - t0) / 60)


def cmd_plot(a):
    edges = pd.read_csv(os.path.join(a.outdir, "network_edges.csv"))
    conf = pd.read_csv(os.path.join(a.outdir, "consensus_interactions.csv.gz"))
    conf["confident"] = conf["confident"].astype(bool)
    make_all_plots(edges, conf, os.path.join(a.outdir, "plots"), a.plot_format, a.plot_min_edge, a.plot_top_edges,
                   a.cascade_source, a.max_hops, a.plot_top_lr)


def cmd_fake(a):
    make_fake_adata(a.cpdb_dir, a.out, a.cells_per_type, a.n_noise_genes, a.seed)


def main(argv=None):
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    a.func(a)


if __name__ == "__main__":
    main()


