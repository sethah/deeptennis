import numpy as np
import argparse
from pathlib import Path
import logging
from logging.config import fileConfig

from sklearn.metrics.pairwise import pairwise_distances
from hmmlearn.hmm import GaussianHMM

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_cluster_variances(features, assignments, centers):
    variances = []
    for c in range(len(centers)):
        dists = pairwise_distances(features[assignments == c], centers[c].reshape(1, -1))
        variances.append(np.var(dists))
    return variances


"""
python src/models/extract_action.py \
--features-path data/interim/featurized_frames \
--save-path data/processed
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=str)
    parser.add_argument("--save-path", type=str, default=None)

    fileConfig('logging_config.ini')

    args = parser.parse_args()

    save_path = Path(args.save_path) / "action_mask"
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=False)

    features_path = Path(args.features_path)
    files = [f for f in features_path.iterdir() if f.suffix == ".npy"]
    for file in files:
        save_file = save_path / file.name
        if save_file.exists():
            logging.debug(f"{save_file} exists. Skipping.")
            continue
        X = np.load(str(file))
        hmm_n_clusters = 2
        hmm = GaussianHMM(n_components=hmm_n_clusters, covariance_type="diag")
        hmm.fit(X)
        hmm_preds = hmm.predict(X)
        variances = get_cluster_variances(X, hmm_preds, hmm.means_)
        court_cluster_id = np.argmin(variances)
        mask = hmm_preds == court_cluster_id
        np.save(str(save_file), mask)

