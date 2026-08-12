# Ajouter « gray + caméra non-centrale (ours) » comme baseline FullCircle piste rttpf
Mis à jour : mardi 11 août 2026, 10:40 (EDT)

## Étape courante
**Terminé.** Table canonique, viewer, docs et métriques de perf sont à jour avec la graine 2.

## En attente de
- **pueue 1668** (viewer, `precompute_metrics.py`) — dernier `ctl.sh rebuild`. Cosmétique :
  la table et le manifeste sont déjà justes, ce job ne refait que les métriques par vue.
  Attente armée en arrière-plan.

## Fait
- [x] 9 scènes `refit_rttpf` entraînées avec `--camera_opt noncentral` (recette identique à
      la ligne `gray` publiée, un seul flag ajouté).
- [x] Contrôle `--camera_opt off` sur **les 9** scènes : moyenne 28.332 contre 28.328 pour
      le gray publié → le worktree reproduit la baseline, comparaison appariée valide.
- [x] `gray-nc` / `gray-nc-off` ajoutés à `dataset/fullcircle_code/masked_eval_rttpf.py`.
- [x] Fusion dans la table canonique `fullcircle_tracks/rttpf_masked.csv` (54 → 63 lignes,
      9 ajoutées, 0 remplacée) via un stem side + `merge_masked_metrics.py`.
- [x] Viewer : `gray-non-central` câblé, `ctl.sh rebuild` passé, métriques par vue
      pré-calculées (job 1538 Success). Visible sur les 9 scènes `-rttpf`.
- [x] Analyse « pourquoi ça marche sur myscenes et pas ici » :
      `scripts/analysis/calib_consistency.py` + `residual_expressible.py`.
- [x] Sonde LR ×10 (`room2`, job 1524) : exclut l'artefact de schedule.
- [x] `PROTOCOL.md` + `IMPLEMENTATION.md` à jour. Mémoire : `noncentral-camera-model.md`.
- [x] **Seconde graine** : 7 scènes relancées sur carte libre (1539–1545). Écart
      graine-à-graine −0.115 à +0.048 (écart-type 0.046), **plus grand que l'effet mesuré**.
- [x] **Perf propre** (1546, balayage unique sur gpu1, gray inclus qui n'avait aucun
      `fps.csv`) : `noncentral` 10.2 min / 245.6 FPS contre `off` 4.6 min / 419.4 FPS.
- [x] Ré-éval graine 2 → fusion (9 remplacées, 0 ajoutée) → `ctl.sh rebuild`.
- [x] `scripts/fullcircle_rttpf_table.py` : table complète 6 métriques × 5 méthodes →
      `dataset/fullcircle_baselines/fullcircle_rttpf_results.{md,csv}`.

## À faire
- rien

## Décisions et surprises
- **Le résultat est un zéro, et c'est le livrable.** **+0.009 dB** sur le contrôle apparié,
  moyenné sur deux graines (graine 1 : +0.016, graine 2 : +0.003), pour **×2.3 le temps
  d'entraînement et ×0.58 la vitesse de rendu**. La dispersion graine-à-graine (0.046) est
  plus grande que l'effet. SPaGS garde la piste avec 28.678 contre 28.335. Ne pas présenter
  cette ligne comme un gain.
- **Le contrôle `off` reproduit gray sur les trois axes** : PSNR +0.003, gaussiennes à
  0.1–1 %, FPS 419.4 contre 419.9. Le worktree est neutre, la comparaison est valide.
- **Le rung n'est pas cassé** : profil par anneaux croissant vers la périphérie comme la
  théorie l'exige, simplement 35× plus faible que sur `workshop`.
- **Pourquoi** : le +0.33 de myscenes était à **98.6 % une correction que rttpf pouvait
  déjà exprimer** — mauvais ajustement COLMAP, pas classe de modèle insuffisante. Le
  re-bundle de FullCircle (1.102 → 0.866 px) a fait ce travail hors ligne. Reste ~40 % de
  non-centralité vraie, mais sur un run / une scène à ~2× le bruit : ne pas sur-vendre.
- **Le prédicteur à retenir** : le **biais partagé** entre calibrations indépendantes du
  même verre (0.39 px myscenes contre 0.04 px FullCircle), PAS la variance entre elles —
  le verre 2 de FullCircle est aussi dispersé que myscenes et ne gagne rien. Sous ~0.05 px,
  inutile d'entraîner : gray tire un rayon par pixel, sans anti-aliasing.
- **Piège pueue, coûteux** : un groupe pueue **ne réserve pas** la carte. 4 runs sont morts
  en OOM 13 s après le départ parce qu'un process hors groupe tenait 12 GB de la TITAN RTX.
  Garde-fou VRAM ajouté à `scripts/queue_fullcircle_rttpf.sh`. Corollaire : **tout FPS ou
  temps mesuré sur carte partagée est à jeter** — le premier lot donnait 63 à 245 FPS sur
  des scènes comparables.
- **Bug corrigé dans `radial_eval.py`** : il appliquait le masque de la caméra 1 à toutes
  les vues d'un rig multi-caméras. 0.36 % du cadre sur FullCircle, précisément au bord où
  le résidu agit. Lit maintenant `masks.json`. Régression vérifiée : `disk(view)` sur
  `out/tunnel_fisheye_baseline` vaut toujours exactement 28.537.
- **Piège d'unités** : les pixels de COLMAP sont au format capteur (fx 1240), gray tourne à
  `-r 4` (fx 310). Tout chiffre en px doit être à la résolution d'entraînement, sinon il
  est 4× trop grand. Les deux scripts d'analyse le font désormais explicitement.
- **Ce worktree est partagé** avec une autre session qui y commite et y modifie
  `gray/raytracer.py`. Vérifié que son changement du 7 août 14:43 (clé de cache
  `base_bearings`) n'affecte pas ces runs : un seul `--eval-models`, uid distincts, et
  l'écart `psnr.csv` → `results.json` est identique entre les deux lots.
