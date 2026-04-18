# Strava GPX Poster Generator

Genere automatiquement une affiche de parcours (style carte + statistiques + profil d'elevation) a partir d'un fichier GPX exporte de Strava.

## Fonctionnalites

- Lecture d'un fichier GPX (`trkpt`) 
- Calcul automatique:
  - Distance totale
  - Denivele positif
  - Duree (si horodatage present) ou estimation via vitesse moyenne
  - Vitesse moyenne
- Rendu d'une affiche PNG avec:
  - Fond carte (tuiles OSM)
  - Trace coloree
  - Points depart/arrivee
  - Titre vertical
  - Bloc de stats
  - Profil d'altitude
- Theme parametrique via JSON:
  - Couleurs
  - Tailles
  - Police custom (`font_path`)
  - Style map design (desaturation, vignette, grain, glow)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
python poster_generator.py \
  --gpx /chemin/vers/trace.gpx \
  --title "Kokopelli Trail" \
  --subtitle "Moab, Utah" \
  --date "2026-04-18" \
  --output poster.png \
  --theme theme.example.json
```

### Ajouter une textbox (taille en cm)

```bash
python poster_generator.py \
  --gpx /chemin/vers/trace.gpx \
  --title "Kokopelli Trail" \
  --output poster.png \
  --theme theme.example.json \
  --textbox-text "Mon texte personnalise" \
  --textbox-width-cm 9 \
  --textbox-height-cm 4.5 \
  --textbox-font-path /chemin/vers/ma-police.ttf
```

Si `--textbox-x-cm` et `--textbox-y-cm` ne sont pas fournis, la box est centree automatiquement.

### Options utiles

- `--avg-speed-kmh 15.5`: utilise cette vitesse pour estimer la duree si le GPX n'a pas de timestamps
- `--tile-zoom 10`: force le zoom carte (sinon auto)
- `--no-tiles`: desactive le telechargement de tuiles et utilise un fond uni

## Exemple de theme

Tu peux partir de `theme.example.json`, le copier puis modifier:

- `typography.font_path`: chemin vers un fichier `.ttf` ou `.otf`
- `map.route_color`, `poster.background`, etc.
- `map.style.desaturate`: 0.0 a 1.0 (carte plus ou moins desaturee)
- `map.style.tint_color` + `map.style.tint_strength`: teinte generale
- `map.style.vignette_strength`: assombrit legerement les bords
- `map.style.grain_strength`: texture papier subtile
- `map.route_glow_width`: halo autour de la trace

Tu peux aussi regler le style de la textbox dans le theme (`textbox`):

- `width_cm`, `height_cm`, `x_cm`, `y_cm`
- `padding_cm`
- `font_path`, `min_font_size`, `max_font_size`
- `align` (`left`, `center`, `right`)
- `bg_color`, `border_color`, `border_width`, `text_color`

## Notes

- Les tuiles OpenStreetMap necessitent une connexion Internet.
- Respecte les conditions d'utilisation du fournisseur de tuiles.
