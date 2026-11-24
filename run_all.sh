set -e
PY=${PY:-python}
$PY train_ae_family.py
$PY mlp_proof.py
$PY train_gan.py gan
$PY train_gan.py dcgan
$PY train_ddpm.py
$PY evaluate.py