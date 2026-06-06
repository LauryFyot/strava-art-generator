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
  - Galerie photo optionnelle
  - Cartes d'information (meteo, activite, altitude max)
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

### Variante poster editorial, proche de l'exemple fourni

```bash
python poster_generator.py \
  --gpx gpx/Fiou.gpx \
  --title "YORKSHIRE PEAKS" \
  --subtitle "Yorkshire Dales" \
  --location "Yorkshire Dales" \
  --date "2026-06-06" \
  --weather "Light showers" \
  --activity "Trail running" \
  --photo ./photo-1.jpg \
  --photo ./photo-2.jpg \
  --photo ./photo-3.jpg \
  --output artifacts/yorkshire_editorial.png \
  --theme theme.example.json
```

Si `--textbox-x-cm` et `--textbox-y-cm` ne sont pas fournis, la box est centree automatiquement.

### Options utiles

- `--avg-speed-kmh 15.5`: utilise cette vitesse pour estimer la duree si le GPX n'a pas de timestamps
- `--tile-zoom 10`: force le zoom carte (sinon auto)
- `--no-tiles`: desactive le telechargement de tuiles et utilise un fond uni
- `--photo /chemin/image.jpg`: ajoute une photo dans la galerie du bas, a repeter jusqu'a 3 fois
- `--weather "Light showers"`: texte de la carte meteo
- `--activity "Trail running"`: texte de la carte activite
- `--location "Yorkshire Dales"`: petit label de lieu au-dessus du bloc stats

## Exemple de theme

Tu peux partir de `theme.example.json`, le copier puis modifier:

- `typography.font_path`: chemin vers un fichier `.ttf` ou `.otf`
- `map.route_color`, `poster.background`, etc.
- `map.style.desaturate`: 0.0 a 1.0 (carte plus ou moins desaturee)
- `map.style.tint_color` + `map.style.tint_strength`: teinte generale
- `map.style.vignette_strength`: assombrit legerement les bords
- `map.style.grain_strength`: texture papier subtile
- `map.route_glow_width`: halo autour de la trace
- `layout.gallery_height`: hauteur de la galerie photo
- `layout.facts_height`: hauteur de la rangee profile + cartes info
- `gallery.*`: style de la galerie photo
- `facts.*`: style des cartes d'information

Tu peux aussi regler le style de la textbox dans le theme (`textbox`):

- `width_cm`, `height_cm`, `x_cm`, `y_cm`
- `padding_cm`
- `font_path`, `min_font_size`, `max_font_size`
- `align` (`left`, `center`, `right`)
- `bg_color`, `border_color`, `border_width`, `text_color`

## Notes

- Les tuiles OpenStreetMap necessitent une connexion Internet.
- Respecte les conditions d'utilisation du fournisseur de tuiles.
