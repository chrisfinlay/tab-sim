# Time-blocked observation geometry

Antenna UVW now stays in the existing Dask graph, with the fine-time chunk size
chosen by the strict planner. All antennas stay together because the kernel uses
antenna zero as its coordinate origin. `no_w` remains a lazy operation. The
synthesized beam-width estimate evaluates only the first time block, rather than
materializing every antenna/time coordinate first.

Skyfield antenna frame transforms and resolved TLE/OMM propagation also execute
one time block at a time. `itrs_to_gcrs_blocks` and `satellite_position_blocks` in
`tabsim.dask.coordinates` accept chunked one-dimensional Julian dates and return
lazy arrays. Explicit shape/type metadata prevents propagation during graph
construction. Orbit records are copied when constructing the graph; tasks perform
no catalogue search, cache resolution or network acquisition.

The existing `ts.ut1_jd` convention, Skyfield transformations, metre units,
source ordering and fine-time integration indexing are unchanged. Offline replay
uses the same frozen records as before. This is a memory-lifetime change, not a
new astrometric approximation.

The one-dimensional time, sidereal-angle and altitude/azimuth vectors remain
eager where previously prepared. The change removes the much larger
`time × antenna × 3` and `source × time × 3` eager histories. It does not claim
constant total setup memory as observation duration grows: time vectors and Dask
graph metadata still grow. The chunk working-set model's exclusions for graph
and runtime allocations remain applicable.

Independent component and ancillary consumers can recompute geometry blocks.
That is an explicit memory/compute tradeoff; neither a full-history `persist()`
nor an unbounded geometry cache is introduced. Use output selection to omit
unneeded fine-time ancillary arrays. Compare setup RSS separately from complete
write time when assessing this change.
