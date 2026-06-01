# README - Unterschiede PyPSA/PyPSA-Eur (Atlite) vs compute_profiles.py

Dieses Dokument beschreibt die Unterschiede zwischen der Atlite-Nutzung in
PyPSA/PyPSA-Eur und dem lokalen Skript `compute_profiles.py` in diesem Ordner.
Der Fokus liegt auf Maskierung, Inputs, Outputs und der Bus-Zuordnung.

## 1) Ziel und Scope

PyPSA-Eur:
- Pipeline (Snakemake) zur Erzeugung von erneuerbaren Profilen und Potenzialen.
- Atlite wird genutzt, um CFs und Verfuegbarkeiten bus-scharf abzuleiten.
- Starke Kopplung an das PyPSA-Eur Datenlayout (Netz, Layouts, Clustering).

compute_profiles.py:
- Standalone-Skript zur Berechnung von CF-Rastern und Verfuegbarkeiten.
- Kann PyPSA-Eur Defaults nachbilden, bietet aber zusaetzliche Optionen.
- Bus-scharfe CFs werden nicht direkt erzeugt; dafuer gibt es
  Availability-Matrizen.

## 2) Inputs und Datenquellen

Gemeinsam (typisch):
- Atlite Cutouts (`cutout_dir/europe-{year}.nc`).
- Raster fuer Landnutzung (CORINE), Natura, Shipdensity, Bathymetrie (GEBCO).
- Vektor-Shapes fuer Land/Wasser, Laender, Regionen.

Nur compute_profiles.py (zusatzlich/optional):
- LUISA Landnutzung (inkl. Legende).
- WDPA (Polygone + Punkte) fuer Offshore und optional Onshore.
- Separierte Mask-Shapes vs Regions-Shapes (z.B. EEZ fuer Offshore).

Hydro (nur compute_profiles.py):
- `plants.csv` + `buses.csv` aus `NETWORK_DIR`.
- Landfaecher aus `country_shapes.geojson`.
- Optional: Normalisierung gegen `hydro_country_profiles.nc`.

## 3) Maskierung / Exclusions

PyPSA-Eur (typisch):
- CORINE Landnutzung (PV/Onshore unterschiedliche Klassen).
- Natura2000.
- Urban-Buffer fuer Onshore (1 km).
- Offshore: Bathymetrie (max depth), Kuestendistanz (min/max), Shipdensity.
- Keine WDPA-Nutzung im aktuellen Repo-Stand.

compute_profiles.py:
- ExclusionContainer (hochauflosend) oder direkte Rastermasken.
- CORINE und/oder LUISA; Fusion moeglich:
  - `intersection` (AND)
  - `prefer-corine`
  - `prefer-luisa` (Default)
- Natura, Shipdensity, Bathymetrie, Kuestendistanzen wie PyPSA-Eur.
- WDPA Offshore (`exclude_wdpa_offshore`) und optional strenge Onshore-Filter
  (`exclude_wdpa_onshore`).
- Maskierte CF-Zellen werden auf NaN gesetzt (statt 0).

## 4) Offshore Defaults und Varianten

PyPSA-Eur (AC/DC):
- AC/DC unterscheiden sich u.a. in Kuestendistanz-Filtern.
- Max depth typischerweise 60 m.

compute_profiles.py:
- `pypsa_offwind_variant` = ac / dc / acdc / float.
- `acdc` setzt nur Max-Depth (keine Kuestendistanz).
- Kann damit bewusst von PyPSA-Eur abweichen.

## 5) Availability und Ressourcenklassen

PyPSA-Eur:
- Verfuegbarkeiten werden bus-scharf ueber Layouts/Regions berechnet.
- CFs werden haeufig direkt auf Bus-Ebene aggregiert.

compute_profiles.py:
- Schreibt separate Availability-Dateien:
  - `availability_onshore.nc`
  - `availability_offshore.nc`
- Verfuegbarkeit als Raster oder pro Region (`availability_mode`).
- Optionale Ressourcenklassen (global oder per-region) + `p_nom_max`.
- CF-Raster bleiben `time,y,x`; Bus-Aggregation muss extern erfolgen.

## 6) Outputs

compute_profiles.py:
- CF-Zeitreihen:
  - `pv/pv_cf_{year}.nc`
  - `onwind/onwind_cf_{year}.nc`
  - `offwind/offwind_cf_{year}.nc`
- Availability:
  - `availability_onshore.nc`, `availability_offshore.nc`
  - `area_km2`, `availability_*`, `p_nom_max_*`, optional Klassen/Bins
- Hydro:
  - `hydro/hydro_bus_profiles_{year}.nc`
  - `hydro/hydro_country_profiles_{year}.nc`

PyPSA-Eur:
- Bus-scharfe Profile und Layouts im Pipeline-Format (Snakemake Outputs).

## 7) Bus-Zuordnung

compute_profiles.py:
- RES: Bus-Zuordnung nur ueber Availability-Matrizen (regions/offshore).
- Hydro: Bus-Zuordnung direkt aus `plants.csv` + `buses.csv` (Landzuordnung).

PyPSA-Eur:
- RES und Hydro sind in der Pipeline auf Bus-/Cluster-Ebene integriert.

## 8) Wichtige Abweichungen (Kurzliste)

- LUISA-Unterstuetzung und Landuse-Fusion (nicht PyPSA-Eur Standard).
- WDPA Offshore/Onshore (nicht PyPSA-Eur Standard).
- EEZ als Offshore-Maskenquelle (falls konfiguriert).
- CFs als Raster (nicht bus-scharf), Availability separat.
- Maskierte CF-Zellen -> NaN.
- YAML-Konfig + `masks_out_only`.
- Turbinen-Check (frueh): Bei `masks_out_only` wird die OEDB/atlite-Turbinenpruefung uebersprungen.
- Optionales `clip_p_max_pu`: setzt kleine CF-Werte auf 0 (PV/onwind/offwind), auch ohne Bus-Zuordnung.

## 9) Wie man PyPSA-Eur Defaults nahe kommt

Empfohlene Settings:
- `pypsa_defaults: true`
- `landuse_dataset: corine`
- `exclude_wdpa_offshore: false`
- `exclude_wdpa_onshore: false`
- `pypsa_offwind_variant: ac` oder `dc` (je nach Ziel)
- `urban_distance_onwind` wird bei `pypsa_defaults` automatisch auf 1000 m gesetzt
- `shipdensity_threshold` wird bei `pypsa_defaults` auf PyPSA-Wert gesetzt

Damit sind Maskierung und Schwellenwerte sehr nah an PyPSA-Eur,
mit dem Unterschied, dass compute_profiles.py weiterhin Raster-CFs ausgibt.

## 10) Konkretes YAML-Beispiel (regions, maskiert)

```yaml
start_year: 1982
end_year: 2024
cutout_dir: '//IIP-COMP103/endata/MA_Lisa/atlite_cutouts/cutouts'
out_dir: '//IIP-COMP103/endata/MA_Eric/res_masked_regions'

resources: [pv, onwind, offwind]

mask_onshore: true
mask_offshore: true
use_excluder_mask: true
excluder_res_onshore: 100
excluder_res_offshore: 200

onshore_mask_shapes:
  - 'Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf\datashapes\europe_shape.geojson'
offshore_mask_shapes:
  - 'Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf\datashapes\eez_offshore_eu27_uk_no_europe_only.geojson'
onshore_regions: 'Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf\datashapes\regions_onshore_bus_2030_voronoi.geojson'
offshore_regions: 'Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf\datashapes\eez_offshore_eu27_uk_no_europe_only.geojson'

landuse_dataset: both
landuse_fusion: prefer-luisa
corine_landcover_path: 'Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf\datashapes\CORINE_LANDCOVER_U2018_CLC2018_V2020_20u1.tif'
luisa_landcover_path: 'Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf\datashapes\LUISA_basemap_020321_100m.tif'

pypsa_defaults: true
pypsa_offwind_variant: acdc
exclude_wdpa_offshore: true
exclude_wdpa_onshore: true
shipdensity_threshold: 1e7

availability_mode: regions
resource_class_mode: per-region
resource_classes: 4
resource_class_year: 2024

capacity_per_sqkm_pv: 6
capacity_per_sqkm_onwind: 3.5
capacity_per_sqkm_offwind: 2.5
```

## 11) Checklisten pro Workflow

Workflow A: Maskierte RES + Availability (regions)
- `resources: [pv, onwind, offwind]`
- `mask_onshore: true`, `mask_offshore: true`, optional `use_excluder_mask: true`
- `onshore_regions` und `offshore_regions` gesetzt
- `availability_mode: regions`, `resource_class_mode: per-region`
- Ausgaben: `availability_onshore.nc`, `availability_offshore.nc` + CF-Raster

Workflow B: Maskierte RES + Availability (raster)
- Wie A, aber `availability_mode: raster`
- `resource_class_mode: global` (oder weglassen)
- Ausgaben: Availability als Raster (keine bus-scharfen Aggregationen)

Workflow C: Unmaskierte RES (Raster-CF)
- `mask_onshore: false`, `mask_offshore: false`, `use_excluder_mask: false`
- Optional: `pypsa_defaults: false` (verhindert Mask-Warnungen)
- Ausgaben: nur CF-Raster ohne NaN-Masken

Workflow D: Hydro-only (bus + country)
- `resources: [hydro]`
- `mask_onshore/offshore` egal
- Stelle sicher, dass `NETWORK_DIR/plants.csv` und `buses.csv` passen
- Ausgaben: `hydro/hydro_bus_profiles_YYYY.nc` und `hydro/hydro_country_profiles_YYYY.nc`

Workflow E: Nur Masken/Availability (keine CFs)
- `masks_out_only: true`
- `mask_onshore/offshore` wie gewuenscht
- Ausgaben: Availability-Dateien, keine CF-Zeitreihen
