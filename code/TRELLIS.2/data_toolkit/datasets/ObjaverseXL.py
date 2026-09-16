import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import objaverse.xl as oxl
from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--source', type=str, default='sketchfab',
                        help='Data source to download annotations from (github, sketchfab)')


def get_metadata(source, **kwargs):
    if source == 'sketchfab':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
    elif source == 'github':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_github.csv")
    else:
        raise ValueError(f"Invalid source: {source}")
    return metadata
        

def download(metadata, output_dir=None, download_root=None, **kwargs):
    output_dir = output_dir or download_root
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    # download annotations
    annotations = oxl.get_annotations()
    annotations = annotations[annotations['sha256'].isin(metadata['sha256'].values)]
    
    # download and render objects
    file_paths = oxl.download_objects(
        annotations,
        download_dir=os.path.join(output_dir, "raw"),
        save_repo_format="zip",
    )
    
    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    for k, v in file_paths.items():
        sha256 = metadata.loc[k, "sha256"]
        downloaded[sha256] = os.path.relpath(v, output_dir)

    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])

#EDIT- no_file=False added to accept the keyword TRELLIS.2 dual_grid/voxelize drivers pass;
# default to false preserves original behavior for dump/download stages (which never 
# pass the kwarg [keyword args]) -> no_file = True stages operate on already dumped data
# so no file resolution applies
def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects', no_file=False) -> pd.DataFrame:    
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    import tempfile
    import zipfile
    
    # load metadata
    metadata = metadata.to_dict('records')

    # processing objects
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
            def worker(metadatum):
                try:
                    #EDIT- implements added mode for stages that operate on previously dumped data
                    # via prebound roots in callback where no file resolution applies 
                    # record=func(metadatum) there for predumped files to fill that record + return
                    if no_file:
                        record = func(None, metadatum) #_dual_grid_mesh and  _pbr_voxelize require an input in the "file" position: 
                                                       #14:def _dual_grid_mesh(file, metadatum, mesh_dump_root, root): [from dual_grid.py]
                                                       #15:def _pbr_voxelize(file, metadatum, pbr_dump_root, root): [from voxelize_pbr.py]
                        if record is not None:
                            records.append(record)
                        pbar.update()
                        return
                    
                    local_path = metadatum['local_path']
                    sha256 = metadatum['sha256']
                    if local_path.startswith('raw/github/repos/'):
                        path_parts = local_path.split('/')
                        file_name = os.path.join(*path_parts[5:])
                        zip_file = os.path.join(output_dir, *path_parts[:5])
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                zip_ref.extractall(tmp_dir)
                            file = os.path.join(tmp_dir, file_name)
                            #FIX– trellis2 expects a sha256 metadata dict, but prior trellis1
                            # provides a sha256 STRING -> 
                            # change:
                            # record = func(file, sha256) [sha256 string]
                            # TO
                            # record = func(file, metadatum) [sha256 metadata]
                            record = func(file, metadatum)
                    else:
                        file = os.path.join(output_dir, local_path)
                        record = func(file, metadatum)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                #EDIT- exception handler edited because old handler references variable assigned
                # only on filepath; exception in no_file branch would NameError (exception that is raised when the program tries to access a variable, function, 
                # or module that has not been defined or initialized) and mask true error
                except Exception as e:
                    print(f"Error processing object {metadatum.get('sha256', '?')}: {e}")
                    pbar.update()
            
            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    return pd.DataFrame.from_records(records)
