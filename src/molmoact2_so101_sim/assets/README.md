# Bundled assets

## wood_table.png

The table texture used by `molmoact2_so101_sim.realism`. It is **original procedural work**,
generated from numpy noise by [`tools/generate_wood_texture.py`](../../../tools/generate_wood_texture.py)
with no third-party source image. It carries no embedded metadata and is distributed under
this repository's Apache-2.0 license.

Regenerate it with:

```bash
python tools/generate_wood_texture.py src/molmoact2_so101_sim/assets/wood_table.png
```

`realism.py` loads this file through `importlib.resources`, so it resolves in a non-editable
install. Set `MOLMOACT2_SO101_TEXTURE_DIR` to a directory holding your own `wood_table.png`
to override it.
