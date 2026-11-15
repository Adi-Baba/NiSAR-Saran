# SAR interpretation figures

These figures are generated from the two local NISAR L2 GCOV frequency-B rasters.
They use display stretches and false-colour channel assignments similar to common
SAR interpretation products. They are not optical imagery, and none of them is a
flood-inundation classification.

The presentation approach follows NASA Earthdata SAR image-interpretation guidance:
https://www.earthdata.nasa.gov/learn/earth-observation-data-basics/sar/image-interpretation
Each false-colour PNG states its exact channel assignment in the title.

- 01–04: matched before/after HHHH and HVHV intensity views.
- 05–07: bitemporal change and polarimetric contrast.
- 08–09: analytical false-colour composites.
- 10: extreme positive-change candidates and selected hotspot.
- 11–13: hotspot zoom and quality layers.

Regenerate with: python3 make_sar_interpretation_figures.py
