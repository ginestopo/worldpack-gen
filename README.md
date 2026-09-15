# worldpackgen

Generador del formato **Worldpack v1** de Trappet, pensado para ejecutarse en
GitHub Actions sin descargar las fuentes raster completas.

## Uso en GitHub

1. Sube este repo a GitHub (público, para que los minutos de Actions sean gratis).
2. Pestaña **Actions** → `build-pack` → **Run workflow** → elige la región.
3. Al terminar, el `.pack` y `build_report.json` aparecen en **Releases**
   (y como artefacto del run durante 7 días).

Jobs: `test` → `osm` (descarga y filtra el extracto de Geofabrik) →
`shard` ×4 en paralelo (lee WorldCover y DEM por ventanas desde AWS) →
`assemble` (dedup, CRC, release).

## Uso local

```bash
pip install -r requirements.txt
python -m pytest -q tests
python -m worldpack.cli query poland.pack 21.0122 52.2297
```

## Estado respecto al spec

- Formato binario completo: header, directorio raíz, sub-índices, seis capas,
  seis modos, POIs, dedup y CRC.
- Bioma solo desde ESA WorldCover (sin CORINE todavía): bosque mixto,
  matorral de transición, pastizal, cultivo herbáceo, urbano denso/disperso,
  roca, glaciar, humedal, tundra, mar y lago.
- Construido: fracción de subpíxeles urbanos de WorldCover (en lugar de GHSL).
- Agua fina, sendas, zonas protegidas y POIs desde OSM.
- Altitud negativa recortada a 0.
- El bbox es rectangular: WorldCover y DEM cubren también los países vecinos,
  pero OSM solo la región del extracto.
- Test 4 del capítulo 10 (mmap vs pread) queda para cuando exista `worldpack.c`.

## Licencias de las fuentes

ESA WorldCover (CC BY 4.0), Copernicus DEM GLO-30 (licencia Copernicus),
OpenStreetMap (ODbL, requiere atribución).
