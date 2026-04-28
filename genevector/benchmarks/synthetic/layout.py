"""
Spatial synthetic dataset generator.

Vendored from grafiti (https://github.com/nceglia/grafiti) — layout and
marker-assignment logic only. Preprocessing and plotting helpers are
intentionally excluded.
"""

import numpy as np
import pandas as pd
import anndata
import random


def generate_synthetic_data(
    num_cells: int = 10000,
    tumor_fraction: float = 0.6,
    stromal_fraction: float = 0.1,
    t_fraction: float = 0.1,
    b_fraction: float = 0.1,
    myeloid_fraction: float = 0.1,
    aggregates_fraction: float = 0.5,
    infiltrated_fraction: float = 0.1,
    scattered_tumor_fraction: float = 0.05,
    tumor_clustered_fraction: float = 0,
    border_width: int = 5,
    num_markers: int = 15,
    num_tumor_gradient: int = 5,
    rare_t_cell: bool = True,
    seed: int | None = None
):

    """
    Simulates a real-world cancer dataset (one field of view).

    Parameters:
        num_cells (int): total number of cells in image
        tumor_fraction (float): fraction of total cells that represent tumor cells
        stromal_fraction (float): fraction of total cells that represent stromal cells
        t_fraction (float): fraction of total cells that represent T cells
        b_fraction (float): fraction of total cells that represent B cells
        myeloid_fraction (float): fraction of total cells that represent myeloid cells
        aggregates_fraction (float): fraction of B cells in an aggregate (clustered) [aggregate size varies between 100-200 based on
            other user-defined params]
        infiltrated_fraction (float): fraction of immune cells (T, B, myeloid) infiltrated in the tumor region [border cells do not
            contribute to this value]
        scattered_tumor_fraction (float): fraction of tumor cells in stromal region
        tumor_clustered_fraction (float): fraction of clustered tumor cells in tumor region
        border_width (int): width of tumor-immune boundary (cannot be greater than 10) [image dimensions are 0-100]
        num_markers (int): total number of markers [to be evenly distributed amongst 5 cell types, unequal distrubtions automatically
            assigned to tumor]
        num_tumor_gradient (int): number of spatial layers (gradients) of expression of marker 0 (tumor marker) in the tumor region
        rare_t_cell (bool): if True the expression for the first T cell marker is between 0.9-1 for 10% of "rare" T cells and 0.7-0.8
            for "non-rare" T cells
        seed (int/None): set seed to a number for reproducibility

    Returns:
        df: dataframe containing cell (rows) by marker, coordinates, features (columns)
        marker_assignemnts: dictionary containing cell type to marker assignments
    """

    # --- Confirm input fractions total to 1 ---
    total_fraction = tumor_fraction + t_fraction + b_fraction + myeloid_fraction + stromal_fraction
    if not np.isclose(total_fraction, 1.0):
        raise ValueError("Total cell type fractions must equal 1.")

    # --- Confirm border_width is less than or equal to 10 ---
    if border_width > 10:
        raise ValueError("border_width must be less than or equal to 10.")

    # --- Set seed for reproducibility ---
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # --- Define cell types and immune subset ---
    cell_types = ['Tumor', 'T cell', 'B cell', 'Myeloid', 'Stromal']
    immune_types = ['T cell', 'B cell', 'Myeloid']

    # --- Distribute markers among cell types ---
    n_types = len(cell_types)
    markers_per_type = num_markers // n_types
    extra = num_markers % n_types
    marker_assignments = {}
    marker_names = []
    marker_index = 0

    for i, ct in enumerate(cell_types):
        n = markers_per_type + (1 if i < extra else 0)
        assigned = [f'marker_{marker_index + j}' for j in range(n)]
        marker_assignments[ct] = assigned
        marker_names.extend(assigned)
        marker_index += n

    # Assign tumor gradient marker
    tumor_gradient_marker = marker_assignments['Tumor'][0]

    # --- Pick a T cell marker for rare expression (optional) ---
    if rare_t_cell == True:
        t_cell_markers = marker_assignments.get('T cell', [])
        rare_marker = t_cell_markers[0] if t_cell_markers else None
    else:
        rare_marker = None

    # --- Define jagged tumor boundary ---
    def jagged_boundary(y):
        return 50 + 5 * np.sin(y / 8.0) + np.random.normal(0, 1)

    data = []

    # --- Precompute counts for special cell placements ---
    num_infiltrated = int(infiltrated_fraction * (t_fraction + b_fraction + myeloid_fraction) * num_cells)
    num_scattered_tumor = int(scattered_tumor_fraction * tumor_fraction * num_cells)
    infiltrated_count = 0
    scattered_tumor_count = 0

    # --- Calculate number of cells per type ---
    num_cells_by_type = {
        'Tumor': int(num_cells * tumor_fraction),
        'T cell': int(num_cells * t_fraction),
        'B cell': int(num_cells * b_fraction),
        'Myeloid': int(num_cells * myeloid_fraction),
        'Stromal': int(num_cells * stromal_fraction),
    }

    # --- Generate B cell aggregates (spatial clustering) ---
    b_cells_total = num_cells_by_type['B cell']
    b_cells_in_aggregates = int(aggregates_fraction * b_cells_total)
    b_cells_remaining = b_cells_total - b_cells_in_aggregates
    aggregate_coords = []
    b_agg_cells_placed = 0

    while b_agg_cells_placed < b_cells_in_aggregates:
        target = min(random.randint(100, 200), b_cells_in_aggregates - b_agg_cells_placed)
        cx, cy = np.random.uniform(60, 95), np.random.uniform(5, 95)
        radius = np.random.uniform(2, 10)
        cov = [[radius / 2, 0], [0, radius / 2]]
        cluster_points = np.random.multivariate_normal([cx, cy], cov, size=target)

        for pt in cluster_points:
            x, y = pt
            if not (0 <= x <= 100 and 0 <= y <= 100):
                continue

            # Marker expression for B cell in aggregates
            marker_intensities = {
                marker: np.random.uniform(0.7, 1.0) if marker in marker_assignments['B cell']
                else np.random.uniform(0.0, 0.3)
                for marker in marker_names
            }

            boundary_x = jagged_boundary(y)
            distance_from_edge = abs(boundary_x - x)

            row = {
                'x': x,
                'y': y,
                'cell_type': 'B cell',
                'infiltrated': 'N',
                'aggregate': 'Y',
                'tumor_clustered': 'N',
                'rare_t_cell': False,
                'border_cell': 'N',
                'region': 'Stroma',
                'distance_from_boundary': distance_from_edge,
                **marker_intensities
            }

            data.append(row)
            b_agg_cells_placed += 1
            num_cells_by_type['B cell'] -= 1

        aggregate_coords.append((cx, cy))

    # --- Generate clustered tumor cells (optional) ---
    tumor_cells_total = num_cells_by_type['Tumor']
    tumor_clustered_count = int(tumor_clustered_fraction * tumor_cells_total)
    clustered_tumor_cells_placed = 0

    while clustered_tumor_cells_placed < tumor_clustered_count:
        cluster_size = min(random.randint(100, 200), tumor_clustered_count - clustered_tumor_cells_placed)
        cx, cy = np.random.uniform(5, 45), np.random.uniform(5, 95)
        radius = np.random.uniform(2, 10)
        cov = [[radius / 2, 0], [0, radius / 2]]
        cluster_points = np.random.multivariate_normal([cx, cy], cov, size=cluster_size)

        for pt in cluster_points:
            x, y = pt
            if not (0 <= x <= 100 and 0 <= y <= 100):
                continue

            boundary_x = jagged_boundary(y)
            if x >= (boundary_x - border_width):
                continue  # Skip cells too close to stroma boundary

            # Assign marker intensities
            marker_intensities = {
                marker: np.random.uniform(0.7, 1.0) if marker in marker_assignments['tumor']
                else np.random.uniform(0.0, 0.3)
                for marker in marker_names
            }

            # Apply distance-based gradient for one marker
            distance_from_edge = max(0, min(50, boundary_x - x))
            gradient = 1.0 - (distance_from_edge / 50.0)
            if num_tumor_gradient > 1:
                contour_step = 1.0 / (num_tumor_gradient - 1)
                gradient = round(gradient / contour_step) * contour_step
            marker_intensities[tumor_gradient_marker] = gradient

            gradient_index = int((1.0 - gradient) * (num_tumor_gradient - 1) + 1)
            gradient_label = f"A{gradient_index}"

            row = {
                'x': x,
                'y': y,
                'cell_type': 'Tumor',
                'infiltrated': 'N',
                'aggregate': 'N',
                'tumor_clustered': 'Y',
                'rare_t_cell': False,
                'border_cell': 'N',
                'region': 'Tumor',
                'gradient': gradient_label,
                'distance_from_boundary': abs(boundary_x - x),
                **marker_intensities
            }

            data.append(row)
            clustered_tumor_cells_placed += 1
            num_cells_by_type['Tumor'] -= 1

    # --- Place rare T cells along a jagged line within the tumor ---
    if rare_t_cell and num_cells_by_type['T cell'] > 0:
        num_rare = int(0.1 * num_cells_by_type['T cell'])  # 10% rare T cells
        rare_t_cells_placed = 0
        while rare_t_cells_placed < num_rare:
            y = np.random.uniform(30,70)
            center_x = jagged_boundary(y) - border_width - 5  # Safely inside tumor
            x = np.random.normal(center_x, 1.0)  # Slight jitter around jagged line

            if not (0 <= x <= 100 and 0 <= y <= 100):
                continue

            boundary_x = jagged_boundary(y)
            distance_from_edge = abs(boundary_x - x)

            # Marker intensities
            marker_intensities = {
                marker: np.random.uniform(0.0, 0.3)
                for marker in marker_names
            }
            if rare_marker:
                marker_intensities[rare_marker] = np.random.uniform(0.9, 1.0)
            for marker in marker_assignments['T cell']:
                if marker != rare_marker:
                    marker_intensities[marker] = np.random.uniform(0.7, 1.0)

            row = {
                'x': x,
                'y': y,
                'cell_type': 'T cell',
                'infiltrated': 'Y',
                'aggregate': 'N',
                'tumor_clustered': 'N',
                'rare_t_cell': True,
                'border_cell': 'N',
                'region': 'Tumor',
                'gradient': 'NA',
                'distance_from_boundary': distance_from_edge,
                **marker_intensities
            }

            data.append(row)
            rare_t_cells_placed += 1
            num_cells_by_type['T cell'] -= 1
            infiltrated_count += 1

    # --- Generate remaining cells ---
    while sum(num_cells_by_type.values()) > 0:
        x, y = np.random.uniform(0, 100), np.random.uniform(0, 100)
        boundary_x = jagged_boundary(y)

        # Classify region based on x-position
        if x < (boundary_x - border_width):
            region = 'Tumor'
        elif x > (boundary_x + border_width):
            region = 'Stroma'
        else:
            region = 'Boundary'

        # Decide cell type and infiltration based on region
        if region == 'Tumor':
            if x < (boundary_x - border_width):  # True tumor region only
                if infiltrated_count < num_infiltrated:
                    candidates = [ct for ct in immune_types if num_cells_by_type[ct] > 0]
                    if candidates:
                        cell_type = random.choice(candidates)
                        infiltrated = 'Y'
                        infiltrated_count += 1
                    else:
                        cell_type = 'Tumor'
                        infiltrated = 'N'
                else:
                    cell_type = 'Tumor'
                    infiltrated = 'N'
            else:
                cell_type = 'Tumor'
                infiltrated = 'N'
        elif region == 'Stroma':
            if scattered_tumor_count < num_scattered_tumor and num_cells_by_type['Tumor'] > 0:
                cell_type = 'Tumor'
                scattered_tumor_count += 1
                infiltrated = 'N'
            else:
                candidates = [ct for ct in ['T cell', 'B cell', 'Myeloid', 'Stromal'] if num_cells_by_type[ct] > 0]
                if not candidates:
                    continue
                cell_type = random.choice(candidates)
                infiltrated = 'N'
        else:
            candidates = [ct for ct in cell_types if num_cells_by_type[ct] > 0]
            if not candidates:
                continue
            cell_type = random.choice(candidates)
            if region == 'Boundary':
                infiltrated = 'N'  # Border cells are never infiltrated
            elif region == 'Tumor':
                if infiltrated_count < num_infiltrated and cell_type in immune_types:
                    infiltrated = 'Y'
                    infiltrated_count += 1
                else:
                    infiltrated = 'N'
            else:  # stroma region
                infiltrated = 'N'
        if num_cells_by_type[cell_type] <= 0:
            continue

        # Avoid placing non-B cells into B cell aggregates
        in_aggregate = any(np.linalg.norm([x - cx, y - cy]) < 2 for cx, cy in aggregate_coords)
        if in_aggregate and cell_type != 'B cell':
            continue

        # Handle T cells differently if rare_t_cell flag is on (optional)
        marker_intensities = {}
        if cell_type == 'T cell' and rare_t_cell == True:
            rare_flag = False
            for marker in marker_names:
                if marker in marker_assignments['T cell']:
                    marker_intensities[marker] = np.random.uniform(0.7, 1.0)  # Set high initially
                else:
                    marker_intensities[marker] = np.random.uniform(0.0, 0.3)
        else:
            rare_flag = False
            for marker in marker_names:
                if marker in marker_assignments[cell_type]:
                    marker_intensities[marker] = np.random.uniform(0.7, 1.0)
                else:
                    marker_intensities[marker] = np.random.uniform(0.0, 0.3)

        # Compute distance from boundary for all cells
        distance_from_edge = abs(boundary_x - x)

        # Apply tumor marker gradient only for tumor cells
        gradient_label = 'NA'  # Default for non-tumor cells
        if cell_type == 'Tumor':
            gradient = 1.0 - (min(50, distance_from_edge) / 50.0)
            if num_tumor_gradient > 1:
                contour_step = 1.0 / (num_tumor_gradient - 1)
                gradient = round(gradient / contour_step) * contour_step
            marker_intensities[tumor_gradient_marker] = gradient
            if region == 'Stroma':
                gradient_label = 'Stroma'
            else:
                gradient_index = int((1.0 - gradient) * (num_tumor_gradient - 1) + 1)
                gradient_label = f"A{gradient_index}"

        row = {
            'x': x,
            'y': y,
            'cell_type': cell_type,
            'infiltrated': infiltrated,
            'aggregate': 'Y' if cell_type == 'B cell' and in_aggregate else 'N',
            'tumor_clustered': 'N',
            'rare_t_cell': rare_flag,
            'border_cell': 'Y' if region == 'Boundary' else 'N',
            'region': region,
            'gradient': gradient_label,
            'distance_from_boundary': distance_from_edge,
            **marker_intensities
        }

        data.append(row)
        num_cells_by_type[cell_type] -= 1

    df = pd.DataFrame(data)

    # --- Assign rare T cell marker intensities after all data is created (optional) ---
    if rare_t_cell == True:
        if rare_marker:
            df.loc[(df['cell_type'] == 'T cell') & (df['rare_t_cell'] == True), rare_marker] = np.random.uniform(0.9, 1.0,
                df[(df['cell_type'] == 'T cell') & (df['rare_t_cell'] == True)].shape[0])
            df.loc[(df['cell_type'] == 'T cell') & (df['rare_t_cell'] == False), rare_marker] = np.random.uniform(0.7, 0.8,
                df[(df['cell_type'] == 'T cell') & (df['rare_t_cell'] == False)].shape[0])

    # --- Assign combined phenotypes ---
    def assign_phenotype(row):
        ct = row['cell_type']
        r = row['region']
        agg = row['aggregate']
        infl = row['infiltrated']
        grad = row['gradient']
        rare = row['rare_t_cell']

        if ct == 'B cell':
            if infl == 'Y':
                return 'B_infiltrated'
            elif r == 'Boundary':
                return 'B_border'
            elif agg == 'Y':
                return 'B_stroma_aggregate'
            elif r == 'Stroma':
                return 'B_stroma_nonaggregate'

        elif ct == 'T cell':
            if rare is True:
                return 'T_rare'
            elif r == 'Tumor':
                return 'T_infiltrated'
            elif r == 'Boundary':
                return 'T_border'
            elif r == 'Stroma':
                return 'T_stroma'

        elif ct == 'Myeloid':
            if r == 'Tumor':
                return 'Myeloid_infiltrated'
            elif r == 'Boundary':
                return 'Myeloid_border'
            elif r == 'Stroma':
                return 'Myeloid_stroma'

        elif ct == 'Stromal':
            if r == 'Tumor':
                return 'Stromal_infiltrated'
            elif r == 'Boundary':
                return 'Stromal_border'
            elif r == 'Stroma':
                return 'Stromal_stroma'

        elif ct == 'Tumor':
            if grad in {'A1', 'A2', 'A3', 'A4', 'A5'}:
                return f'Tumor_{grad}'
            elif r == 'Stroma':
                return 'Tumor_stroma'
        return 'Unknown'

    df['phenotype'] = df.apply(assign_phenotype, axis=1)

    # --- Reorder dataframe columns ---
    new_order = ['x', 'y', 'phenotype', 'cell_type', 'infiltrated', 'aggregate', 'tumor_clustered',
                 'rare_t_cell', 'border_cell', 'region', 'gradient', 'distance_from_boundary'] + list(marker_intensities.keys())
    df = df[new_order]

    return df, marker_assignments


def create_anndata_from_synthetic(df):

    """
    Creates AnnData object.

    Parameters:
        df (pandas DataFrame): dataframe generated using generate_synthetic_data

    Returns:
        adata (AnnData): containing marker intensities [X], phenotypes [obs], spatial coordinates [obsm]
    """

    features = [col for col in df.columns if col.startswith("marker_")]
    obs_features = [col for col in df.columns if col not in features]

    phenotype_order = ['Tumor_A1', 'Tumor_A2', 'Tumor_A3', 'Tumor_A4', 'Tumor_A5', 'Tumor_stroma',
                       'Stromal_border', 'Stromal_stroma', 'T_infiltrated', 'T_border', 'T_stroma', 'T_rare',
                       'B_infiltrated', 'B_border', 'B_stroma_aggregate', 'B_stroma_nonaggregate',
                       'Myeloid_infiltrated', 'Myeloid_border', 'Myeloid_stroma']
    cell_type_order = ['Tumor', 'Stromal', 'T cell', 'B cell', 'Myeloid']
    region_order = ['Tumor', 'Stroma', 'Boundary']
    df['phenotype'] = pd.Categorical(df['phenotype'], categories=phenotype_order, ordered=True)
    df['cell_type'] = pd.Categorical(df['cell_type'], categories=cell_type_order, ordered=True)
    df['region'] = pd.Categorical(df['region'], categories=region_order, ordered=True)

    adata = anndata.AnnData(X = df[features].to_numpy(),
                            obs = df[obs_features])
    adata.var_names = df[features].columns
    adata.obs['fov'] = pd.Categorical(['1'] * adata.n_obs)
    adata.obsm['spatial'] = adata.obs[['x','y']].to_numpy()

    phenotype_palette = {'Tumor_A1':'#ea9999', 'Tumor_A2':'#e06666', 'Tumor_A3':'#cc0000', 'Tumor_A4':'#990000',
                         'Tumor_A5':'#660000', 'Tumor_stroma':'#f44336', 'Stromal_stroma':'#f9cb9c',
                         'Stromal_border':'#e69138', 'T_infiltrated':'#073763', 'T_border':'#3d85c6',
                         'T_stroma':'#9fc5e8', 'T_rare':'#5aaefa', 'B_infiltrated':'#274e13',
                         'B_border':'#6aa84f', 'B_stroma_aggregate':'#006400', 'B_stroma_nonaggregate':'#b6d7a8',
                         'Myeloid_infiltrated':'#674ea7', 'Myeloid_border':'#8e7cc3', 'Myeloid_stroma':'#b4a7d6'}
    adata.uns['phenotype_colors'] = [phenotype_palette[p] for p in phenotype_order]
    cell_type_palette = {'Tumor':'#eb3542', 'Stromal':'#a98c6c', 'T cell':'#0f92c0', 'B cell':'#5dbf86', 'Myeloid':'#a066a8'}
    adata.uns['cell_type_colors'] = [cell_type_palette[p] for p in cell_type_order]
    region_palette = {'Tumor':'#D62828', 'Stroma':'#008000', 'Boundary':'#003F88'}
    adata.uns['region_colors'] = [region_palette[p] for p in region_order]

    return adata
