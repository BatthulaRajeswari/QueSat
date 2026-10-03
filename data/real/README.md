# Real satellite-derived data

Place real satellite-derived CSV files here (or upload them in the app's **Data Analysis** page).

Required header:

```
Blue,Green,Red,NIR,Change
```

* `Change` must be `0` (No Change) or `1` (Change).
* Optional columns `Latitude`, `Longitude` enable a real coordinate scatter map in the Results page.
* QueSat never downloads satellite data automatically. Export Sentinel-2 (or similar) derived
  band values and labels yourself, then supply them as CSV.

Files in this folder are git-ignored by default.
