from clworkflow import calculate_group_statistics, plot_group_comparison

# Calculate statistics for different groups
yc_stats = calculate_group_statistics(results, group_key='YC')
ad_stats = calculate_group_statistics(results, group_key='AD')

# Visualize group comparison
plot_group_comparison(
    results,
    {'Young Controls': 'YC', 'Alzheimer\'s Disease': 'AD'},
    reference_region='wc',
    outpath=output_dir
)