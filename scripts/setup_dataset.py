#!/usr/bin/env python3
import os
import logging
import argparse
import polars as pl

from tqdm import tqdm
from huggingface_hub import snapshot_download
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

n_procs = max(1, os.cpu_count() - 1)
n_cpus = os.cpu_count()
n_threads = max(1, n_cpus // n_procs)

def setup_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger("IndicvoicesSetup")
    logger.setLevel(log_level)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}\n")

    return logger

def process_row(row, dest_dir):
    try:
        audio_data = row['audio_filepath']['bytes']
        audio_name = row['audio_filepath']['path']
        del row['audio_filepath']

        save_path = os.path.join(dest_dir, audio_name)
        with open(save_path, "wb") as f:
            f.write(audio_data)

        entry = {'audio_filepath': save_path, **row}
        return True, entry
    except Exception as e:
        return False, f"Error processing sample: {e}"

def process_parquet(parquet_file, lang, audio_save_dir, manifest_save_dir, logger):
    index = os.path.basename(parquet_file).split('.')[0]
    dest_dir = os.path.join(audio_save_dir, lang, index)
    os.makedirs(dest_dir, exist_ok=True)

    def log(message):
        logger.info(f"{lang} - {os.path.basename(parquet_file)} - {message}")

    def log_error(message):
        logger.error(f"{lang} - {os.path.basename(parquet_file)} - {message}")

    log("Processing...")
    df = pl.read_parquet(parquet_file)

    manifest = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(process_row, row, dest_dir) for row in df.iter_rows(named=True)]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {lang}/{os.path.basename(parquet_file)}"):
            result = future.result()
            if result[0]:
                manifest.append(result[1])
            else:
                log_error(result[1])

    log(f"Processed {len(manifest)} rows!")

    manifest_df = pl.DataFrame(manifest)
    save_path = os.path.join(manifest_save_dir, f'{lang}_manifests', f"{index}.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    manifest_df.write_ndjson(save_path)
    log(f"Saved manifest to {save_path}")
    return manifest

def main(save_dir: str):
    HF_SAVE_DIR = os.path.join(save_dir, 'hf')
    AUDIO_SAVE_DIR = os.path.join(save_dir, 'audios')
    MANIFEST_SAVE_DIR = os.path.join(save_dir, 'manifests')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(HF_SAVE_DIR, exist_ok=True)
    os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)
    os.makedirs(MANIFEST_SAVE_DIR, exist_ok=True)

    log_file = os.path.join(save_dir, 'setup.log')
    if os.path.exists(log_file):
        os.remove(log_file)
    logger = setup_logger(log_file=log_file)
    logger.info(f"Created directories: {save_dir}, {HF_SAVE_DIR}, {AUDIO_SAVE_DIR}, {MANIFEST_SAVE_DIR}\n")

    logger.info("Downloading IndicVoices from Hugging Face")
    snapshot_download(
        repo_id="ai4bharat/IndicVoices",
        repo_type="dataset",
        local_dir=HF_SAVE_DIR,
        local_dir_use_symlinks=False,
        max_workers=32,
        resume_download=True,
        allow_patterns=["*/valid*.parquet", "*/train-0000[0-5]*.parquet"]
    )
    logger.info(f"Downloaded to {HF_SAVE_DIR}\n")

    logger.info("Processing IndicVoices")
    manifest = []
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        for lang in os.listdir(HF_SAVE_DIR):
            lang_dir = os.path.join(HF_SAVE_DIR, lang)
            if not os.path.isdir(lang_dir):
                continue

            parquet_files = [
                os.path.join(lang_dir, f)
                for f in os.listdir(lang_dir)
                if f.endswith(".parquet")
            ]
            if not parquet_files:
                continue

            logger.info(f"Processing {lang} with {len(parquet_files)} parquet files")
            futures = [
                executor.submit(process_parquet, parquet_file, lang, AUDIO_SAVE_DIR, MANIFEST_SAVE_DIR, logger)
                for parquet_file in parquet_files
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing results"):
                manifest.extend(future.result())

    manifest_df = pl.DataFrame(manifest)
    save_path = os.path.join(MANIFEST_SAVE_DIR, "combined_manifest.jsonl")
    manifest_df.write_ndjson(save_path)
    logger.info(f"Saved combined manifest to {save_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', type=str)
    args = parser.parse_args()
    main(save_dir=args.save_dir)
