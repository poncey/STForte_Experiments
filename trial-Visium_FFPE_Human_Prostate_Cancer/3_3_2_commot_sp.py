import os
os.chdir(path='../')

import datetime
import time

import scanpy as sc
import matplotlib.pyplot as plt

import commot as ct

trial_name = "trial-Visium_FFPE_Human_Prostate_Cancer"
data_name = "Visium_FFPE_Human_Prostate_Cancer"


def convert_seconds_to_hms(duration):
    
    hours = int(duration // 3600)
    
    minutes = int((duration % 3600) // 60)
    
    seconds = int(duration % 60)

    return hours, minutes, seconds

if __name__ == "__main__":
    # Original data
    adata = sc.read_h5ad(f"{trial_name}/outputs/stforte.h5ad")
    adata

    # Find highly variable genes for gene propagation
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3')
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Get adata for original resolution
    adata_dis500_cellchat = adata.copy()
    adata_dis500_cellphonedb = adata.copy()

    # Get dataframe for filtered signals
    df_cellphonedb = ct.pp.ligand_receptor_database(species='human', database='CellPhoneDB_v4.0')
    df_cellchat = ct.pp.ligand_receptor_database(species='human', database='CellChat')
    df_cellphonedb_filtered = ct.pp.filter_lr_database(df_cellphonedb, adata_dis500_cellphonedb, min_cell_pct=0.05)
    df_cellchat_filtered = ct.pp.filter_lr_database(df_cellchat, adata_dis500_cellchat, min_cell_pct=0.05)

    # Make sp data for different databases
    adata_sp = sc.read_h5ad(f"{trial_name}/outputs/sp_genes.h5ad")
    adata_sp_dis500_cellchat = adata_sp.copy()
    adata_sp_dis500_cellphonedb = adata_sp.copy()

    # Generate sp communications with CellChat
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}]\tstart cellchat analysis")
    start_time = time.time()
    
    ct.tl.spatial_communication(adata_sp_dis500_cellchat,
        database_name='cellchat', df_ligrec=df_cellchat_filtered, dis_thr=500, heteromeric=True, pathway_sum=True)
    
    end_time = time.time()
    h, m, s = convert_seconds_to_hms(end_time - start_time)
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}]\tCellChat analysis completed with duration: {h}:{m}:{s}")
    
    # Generate sp communications with CellPhoneDB_v4.0
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}]\tstart cellphonedb analysis")
    start_time = time.time()

    ct.tl.spatial_communication(adata_sp_dis500_cellphonedb,
        database_name='cellphonedb', df_ligrec=df_cellphonedb_filtered, dis_thr=500, heteromeric=True, pathway_sum=True)
    
    end_time = time.time()
    h, m, s = convert_seconds_to_hms(end_time - start_time)
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}]\tCellPhoneDB_v4.0 analysis completed with duration: {h}:{m}:{s}")
    
    # Saving data
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}]\tSaving all data analysed by commot...")
    start_time = time.time()
    adata_sp_dis500_cellchat.write_h5ad(f"{trial_name}/outputs/sp_commot_cellchat.h5ad")
    adata_sp_dis500_cellphonedb.write_h5ad(f"{trial_name}/outputs/sp_commot_cellphonedb.h5ad")
    
    adata_sp_dis500_cellchat.uns['commot-cellchat-info']['df_ligrec'].to_csv(f"{trial_name}/outputs/sp_commot_cellchat.csv")
    adata_sp_dis500_cellphonedb.uns['commot-cellphonedb-info']['df_ligrec'].to_csv(f"{trial_name}/outputs/sp_commot_cellphonedb.csv")
    end_time = time.time()
    h, m, s = convert_seconds_to_hms(end_time - start_time)
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}]\tSaving completed with duration: {h}:{m}:{s}")