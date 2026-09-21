# Retained-output selection

The default schema is unchanged from PR 64: retain all data arrays except
`rfi_*_A`, unless RFI amplitude saving is enabled. Exact `save_arrays` lists keep
their existing meaning. Coordinates and dataset attributes always survive.

Two optional profiles make the common choices explicit:

- `output_profile='full'`: every dataset data variable, including RFI amplitudes.
- `output_profile='minimal'`: `vis_obs`, baseline antenna indices, antenna ITRF
  positions, baseline UVW, fine-time centre indices, noise standard deviations,
  and SEFD. Fine-time antenna/source histories and source amplitudes are omitted.

```python
obs.write_to_zarr('minimal.zarr', output_profile='minimal')
```

For another set of visibility products with the same metadata:

```python
from tabsim.staged import minimal_arrays
obs.calculate_vis()
selection = minimal_arrays(obs.dataset, products=['vis_calibrated', 'flags'])
obs.write_to_zarr('selected.zarr', save_arrays=selection)
```

Use `sim-vis --output-profile minimal ...` or YAML `output.output_profile:
minimal` for the preset. A profile and an exact `save_arrays` list are mutually
exclusive, so an existing exact list is never silently expanded. The full profile
explicitly includes RFI amplitudes regardless of the default amplitude flag.

Composition dependencies are still computed when needed. For example, minimal
observed visibility requires astronomical visibility, RFI visibility, gains and
noise. Those arrays are staged in the same store and deleted after their last
consumer. The profile therefore reduces final size and avoids calibration/flag
and ancillary writes, but still needs scratch for components plus output.
Omitted auxiliary arrays are not separately computed just for writing.

When Zarr and Measurement Set output are both requested, the existing checkpoint
path temporarily retains the MS writer's required inputs, writes the MS, then
prunes the Zarr to the requested selection. Minimal Zarr by itself is not the
full input schema required for a later standalone MS conversion. This change
does not claim to fix the MS-only issue #42.

Optional signal statistics now submit their four scalar reductions together,
allowing common upstream tasks to be shared. They do not persist an observation.
If statistics or MS output request an otherwise omitted product, it is still
computed as a temporary dependency. Disable unnecessary diagnostics for the
smallest workload.

Performance reports must distinguish parent/candidate comparisons with identical
outputs from full/minimal comparisons. Writing fewer products is a deliberate
output tradeoff, not faster identical computation.
