# Ajouter « gray + caméra non-centrale (ours) » comme baseline FullCircle piste rttpf
Mis à jour : lundi 10 août 2026, 16:35 (EDT)

## Étape courante
Rien à faire à la main. Le livrable est en place et vérifiable ; il ne reste qu'à
consommer 8 jobs GPU en file, puis à refaire éval → fusion → `ctl.sh rebuild` et à
compléter la colonne coût.

## En attente de
- **pueue 1539–1545** (Queued, gpu1) — re-runs propres des 7 scènes `noncentral` du lot
  contendu. ~10 min pièce.
- **pueue 1546** (Queued, gpu1) — balayage FPS des 27 runs de la piste, une seule carte,
  d'affilée. ~15 min.
- Devant eux : **1531 (Running) puis 1532–1536**, jobs d'**une autre session**. 1530 a duré
  25 min, donc compter **~2 h 30** avant que 1539 démarre. Ne pas y toucher.
- Une attente en arrière-plan est armée sur 1546.

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

## À faire
- [ ] Quand 1539–1545 sortent : ré-évaluer (`masked_eval_rttpf.py --methods gray gray-nc
      gray-nc-off --out .../rttpf_masked_nc`), fusionner **seulement `gray-nc`**, rebuild.
- [ ] Reporter la seconde graine : écart graine-à-graine sur les 7 scènes = estimation de
      bruit supplémentaire pour le résultat nul.
- [ ] Colonne FPS + temps d'entraînement dans `scripts/fullcircle_rttpf_table.py`
      (script écrit, jamais lancé avec des données propres).

## Décisions et surprises
- **Le résultat est un zéro, et c'est le livrable.** +0.016 dB sur le contrôle apparié
  (écart-type 0.033, positif sur 4 scènes/9) pour **~2.1× le temps d'entraînement**. SPaGS
  garde la piste avec 28.678 contre 28.347. Ne pas présenter cette ligne comme un gain.
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
