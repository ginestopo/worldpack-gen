# worldpackgen

Generador del formato **Worldpack v1** de Trappet, pensado para ejecutarse en
GitHub Actions sin descargar las fuentes raster completas.

## Uso en GitHub

1. Sube este repo a GitHub (público, para que los minutos de Actions sean gratis).
2. Pestaña **Actions** → `build-pack` → **Run workflow** → elige la región.
3. Al terminar, el `.pack` y `build_report.json` aparecen en **Releases**
   (y como artefacto del run durante 7 días).

Jobs: `test` → `osm` (descarga y filtra el extracto de Geofabrik) →
`shard` en paralelo (lee WorldCover y DEM por ventanas desde AWS) →
`assemble` (dedup, CRC, release).

## Regiones

La tabla vive en `REGIONS` (`worldpack/cli.py`): bbox, extracto de Geofabrik,
número de shards y las coordenadas de la comprobación rápida posterior al build.
El workflow lee todo de ahí, así que añadir una región es una entrada más.

| Región | bbox | gh3 | shards |
|---|---|---|---|
| `poland` | 14,49 → 24.2,55 | 54 | 4 |
| `colombia` | −82,−4.3 → −66.8,13.5 | 168 | 12 |
| `zaragoza` | −1.2,41.4 → −0.6,41.8 | 1 | 1 |

El reparto es `ids[shard::shards]`, así que lo que importa es mantener ~14 gh3
por shard: es lo que hace que cada job quepa en el límite de 6 h de Actions.

El bbox de Colombia llega hasta 82° O para incluir San Andrés y Providencia. El
mar sobrante sale casi gratis: un gh3 sin tesela de WorldCover se omite entero.

## Uso local

```bash
pip install -r requirements.txt
python -m pytest -q tests
python -m worldpack.cli query poland.pack 21.0122 52.2297
python -m worldpack.cli query colombia.pack -74.0721 4.7110
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

### Fuera de Europa

Ambas fuentes nombran la tesela por su esquina **suroeste**, así que el
hemisferio sur y el oeste de Greenwich funcionan sin tocar `rasters.py`
(`test_tile_url_hemispheres` y `test_end_to_end_southern` lo fijan). Lo que sí
cambia de contexto:

- **Altitud por encima de 4095 m.** La base del chunk son 12 bits, así que en
  la Sierra Nevada o los nevados se satura y el cuantizador sube el `elev_step`
  para cubrir el rango. No se recorta ningún valor, solo baja la resolución a
  ±16 m en vez de ±2 m. Si eso molesta en el juego, es un cambio de
  `format_version`, no de contenido.
- **Manglar.** WorldCover clase 95 cae hoy en humedal genérico. En Europa daba
  igual; en el Pacífico colombiano es un bioma propio y hay hueco libre en la
  taxonomía (códigos 5, 10, 11, 14, 15, 18–22). Es una línea en
  `WC_TO_BIOME` más un tile de arte, y sube `content_version`.
- **POIs.** `POI_RULES` está escrito para Europa (castillos, calzadas romanas,
  megalitos, molinos, palomares). En Colombia disparan bien volcán, cueva,
  termal, cascada, mirador, yacimiento arqueológico, iglesia y cementerio, y
  quedan vacíos una docena de tipos. El `build_report.json` trae el histograma
  por tipo: conviene mirarlo antes de decidir qué ocupar de los 11 reservados.

## Licencias de las fuentes

ESA WorldCover (CC BY 4.0), Copernicus DEM GLO-30 (licencia Copernicus),
OpenStreetMap (ODbL, requiere atribución).
