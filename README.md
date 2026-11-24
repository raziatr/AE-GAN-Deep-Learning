# Running kode
```
pip install -r requirements.txt
sh download_data.sh          # unduh Fashion-MNIST ke data/
sh run_all.sh                # latih semua model lalu evaluasi (PY=python)
```
Urutan: `train_ae_family.py` → `mlp_proof.py` → `train_gan.py gan|dcgan` → `train_ddpm.py` → `evaluate.py`
Output: `checkpoints/`, `logs/`, `results/results.json`, `results/figs/*.png`

# Standardisasi
Seed 42 (`random`, `numpy`, `torch`), split 80/10/10 (56.000/7.000/7.000) dengan permutasi seed 42, batch 128, shuffle hanya pada train.