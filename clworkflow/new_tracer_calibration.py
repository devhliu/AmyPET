from clworkflow import run_centiloid_pipeline, compare_tracers, plot_centiloid_comparison

# Process PiB data
pib_results = run_centiloid_pipeline(
    pib_pet_files,
    mri_files,
    atlas_dir,
    tracer='pib',
    outpath=output_dir / 'pib'
)

# Process new tracer data
new_results = run_centiloid_pipeline(
    new_pet_files,
    mri_files,
    atlas_dir,
    tracer='new',
    outpath=output_dir / 'new_tracer'
)

# Compare tracers
comparison = compare_tracers(pib_results, new_results, reference_region='wc')

# Visualize comparison
plot_centiloid_comparison(
    comparison['pib_values'],
    comparison['new_values'],
    'New Tracer',
    reference_region='wc',
    outpath=output_dir
)