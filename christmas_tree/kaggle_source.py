"""
Santa 2025 Metric
For each N-tree configuration, calculate the bounding square divided by N.
Final score is the sum of the scores across all configurations.

A scaling factor is used to maintain reasonably precise floating point
calculations in the shapley (v 2.1.2) library.
"""

from decimal import Decimal, getcontext


import pandas as pd
from shapely import affinity, touches
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

# Decimal precision and scaling factor
getcontext().prec = 25
scale_factor = Decimal('1e18')


class ParticipantVisibleError(Exception):
    pass


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))



def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    For each n-tree configuration, the metric calculates the bounding square
    volume divided by n, summed across all configurations.

    This metric uses shapely v2.1.2.

    Examples
    -------
    >>> import pandas as pd
    >>> row_id_column_name = 'id'
    >>> data = [['002_0', 's-0.2', 's-0.3', 's335'], ['002_1', 's0.49', 's0.21', 's155']]
    >>> submission = pd.DataFrame(columns=['id', 'x', 'y', 'deg'], data=data)
    >>> solution = submission[['id']].copy()
    >>> score(solution, submission, row_id_column_name)
    0.877038143325...
    """

    # remove the leading 's' from submissions
    data_cols = ['x', 'y', 'deg']
    submission = submission.astype(str)
    for c in data_cols:
        if not submission[c].str.startswith('s').all():
            raise ParticipantVisibleError(f'Value(s) in column {c} found without `s` prefix.')
        submission[c] = submission[c].str[1:]

    # enforce value limits
    limit = 100
    bad_x = (submission['x'].astype(float) < -limit).any() or \
            (submission['x'].astype(float) > limit).any()
    bad_y = (submission['y'].astype(float) < -limit).any() or \
            (submission['y'].astype(float) > limit).any()
    if bad_x or bad_y:
        raise ParticipantVisibleError('x and/or y values outside the bounds of -100 to 100.')

    # grouping puzzles to score
    submission['tree_count_group'] = submission['id'].str.split('_').str[0]

    total_score = Decimal('0.0')
    for group, df_group in submission.groupby('tree_count_group'):
        num_trees = len(df_group)

        # Create tree objects from the submission values
        placed_trees = []
        for _, row in df_group.iterrows():
            placed_trees.append(ChristmasTree(row['x'], row['y'], row['deg']))

        # Check for collisions using neighborhood search
        all_polygons = [p.polygon for p in placed_trees]
        r_tree = STRtree(all_polygons)

        # Checking for collisions
        for i, poly in enumerate(all_polygons):
            indices = r_tree.query(poly)
            for index in indices:
                if index == i:  # don't check against self
                    continue
                if poly.intersects(all_polygons[index]) and not poly.touches(all_polygons[index]):
                    raise ParticipantVisibleError(f'Overlapping trees in group {group}')

        # Calculate score for the group
        bounds = unary_union(all_polygons).bounds
        # Use the largest edge of the bounding rectangle to make a square boulding box
        side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])

        group_score = (Decimal(side_length_scaled) ** 2) / (scale_factor**2) / Decimal(num_trees)
        total_score += group_score

    return float(total_score)


def score_mod(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> dict:
    """
    For each n-tree configuration, the metric calculates the bounding square
    volume divided by n, summed across all configurations.

    This metric uses shapely v2.1.2.

    Examples
    -------
    >>> import pandas as pd
    >>> row_id_column_name = 'id'
    >>> data = [['002_0', 's-0.2', 's-0.3', 's335'], ['002_1', 's0.49', 's0.21', 's155']]
    >>> submission = pd.DataFrame(columns=['id', 'x', 'y', 'deg'], data=data)
    >>> solution = submission[['id']].copy()
    >>> score(solution, submission, row_id_column_name)
    0.877038143325...
    """
    violates_x_bounds = 0
    violates_y_bounds = 0
    has_intersections = 0

    # remove the leading 's' from submissions
    data_cols = ['x', 'y', 'deg']
    submission = submission.astype(str)
    for c in data_cols:
        if not submission[c].str.startswith('s').all():
            raise ParticipantVisibleError(f'Value(s) in column {c} found without `s` prefix.')
        submission[c] = submission[c].str[1:]

    # enforce value limits
    limit = 100
    bad_x = (submission['x'].astype(float) < -limit).any() or \
            (submission['x'].astype(float) > limit).any()
    bad_y = (submission['y'].astype(float) < -limit).any() or \
            (submission['y'].astype(float) > limit).any()

    if bad_x:
        violates_x_bounds = 1

    if bad_y:
        violates_y_bounds = 1

    # if bad_x or bad_y:
    #     raise ParticipantVisibleError('x and/or y values outside the bounds of -100 to 100.')


    # grouping puzzles to score
    submission['tree_count_group'] = submission['id'].str.split('_').str[0]

    total_score = Decimal('0.0')
    for group, df_group in submission.groupby('tree_count_group'):
        num_trees = len(df_group)

        # Create tree objects from the submission values
        placed_trees = []
        for _, row in df_group.iterrows():
            placed_trees.append(ChristmasTree(row['x'], row['y'], row['deg']))

        # Check for collisions using neighborhood search
        all_polygons = [p.polygon for p in placed_trees]
        r_tree = STRtree(all_polygons)

        # Checking for collisions
        for i, poly in enumerate(all_polygons):
            indices = r_tree.query(poly)
            for index in indices:
                if index == i:  # don't check against self
                    continue
                if poly.intersects(all_polygons[index]) and not poly.touches(all_polygons[index]):
                    has_intersections = 1
                    

        # Calculate score for the group
        bounds = unary_union(all_polygons).bounds
        # Use the largest edge of the bounding rectangle to make a square boulding box
        side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])

        group_score = (Decimal(side_length_scaled) ** 2) / (scale_factor**2) / Decimal(num_trees)
        total_score += group_score

    # return pd.DataFrame([float(total_score),has_intersections, violates_x_bounds, violates_y_bounds],
    #     columns=['score', 'has_intersections', 'violates_x_bounds', 'violates_y_bounds'])
    has_violations = max(has_intersections, violates_x_bounds, violates_y_bounds)
    return {
        'score':float(total_score),
        'has_intersections': has_intersections, 
        'violates_x_bounds': violates_x_bounds, 
        'violates_y_bounds': violates_y_bounds,
        'has_violations': has_violations
    }


def score_with_violations(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, scale_factor: float = 1.0) -> pd.DataFrame:
    """
    Modified version that returns a dataframe with scores and violation flags.
    
    Parameters:
    -----------
    solution : pd.DataFrame
        Solution dataframe with tree IDs
    submission : pd.DataFrame
        Submission dataframe with x, y, deg values
    row_id_column_name : str
        Name of the ID column
    scale_factor : float
        Scaling factor for the score calculation (default: 1.0)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: 
        - 'group': tree group ID
        - 'score': calculated score for the group
        - 'violates_x_bounds': 1 if x values out of bounds, else 0
        - 'violates_y_bounds': 1 if y values out of bounds, else 0
        - 'has_intersections': 1 if trees intersect, else 0
        - 'total_trees': number of trees in the group
        - 'bounding_side': side length of bounding square
    """
    
    # Make a copy to avoid modifying original
    submission_df = submission.copy()
    
    # Check for 's' prefix and remove it
    data_cols = ['x', 'y', 'deg']
    submission_df = submission_df.astype(str)
    
    # Initialize violation columns
    results = []
    
    # Remove 's' prefix if present
    for c in data_cols:
        if submission_df[c].str.startswith('s').all():
            submission_df[c] = submission_df[c].str[1:]
        elif c in ['x', 'y', 'deg']:
            # Check if values are numeric without 's' prefix
            try:
                submission_df[c] = pd.to_numeric(submission_df[c])
            except:
                pass
    
    # Convert to numeric
    for c in ['x', 'y', 'deg']:
        submission_df[c] = pd.to_numeric(submission_df[c], errors='coerce')
    
    # Grouping puzzles to score
    submission_df['tree_count_group'] = submission_df['id'].str.split('_').str[0]
    
    for group, df_group in submission_df.groupby('tree_count_group'):
        num_trees = len(df_group)
        
        # Initialize violation flags for this group
        violates_x_bounds = 0
        violates_y_bounds = 0
        has_intersections = 0
        group_score = 0
        bounding_side = 0
        
        # Check bounds
        limit = 100
        if (df_group['x'].astype(float) < -limit).any() or (df_group['x'].astype(float) > limit).any():
            violates_x_bounds = 1
        
        if (df_group['y'].astype(float) < -limit).any() or (df_group['y'].astype(float) > limit).any():
            violates_y_bounds = 1
        
        # Skip further calculations if bounds are violated
        if violates_x_bounds == 0 and violates_y_bounds == 0:
            try:
                # Create tree objects
                placed_trees = []
                for _, row in df_group.iterrows():
                    placed_trees.append(ChristmasTree(row['x'], row['y'], row['deg']))
                
                # Check for intersections
                all_polygons = [p.polygon for p in placed_trees]
                r_tree = STRtree(all_polygons)
                
                # Check for intersections
                for i, poly in enumerate(all_polygons):
                    indices = r_tree.query(poly)
                    for index in indices:
                        if index == i:  # don't check against self
                            continue
                        if poly.intersects(all_polygons[index]) and not poly.touches(all_polygons[index]):
                            has_intersections = 1
                            break
                    if has_intersections:
                        break
                
                # Calculate score only if no violations
                if has_intersections == 0:
                    bounds = unary_union(all_polygons).bounds
                    side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
                    bounding_side = side_length_scaled
                    
                    if num_trees > 0:
                        group_score = float((Decimal(side_length_scaled) ** 2) / (scale_factor**2) / Decimal(num_trees))
                
            except Exception as e:
                # If any error occurs during processing, mark as having intersections
                has_intersections = 1
        
        # Add to results
        results.append({
            'group': group,
            'score': group_score,
            'violates_x_bounds': violates_x_bounds,
            'violates_y_bounds': violates_y_bounds,
            'has_intersections': has_intersections,
            'total_trees': num_trees,
            'bounding_side': bounding_side
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Calculate overall metrics
    results_df['overall_score'] = results_df['score'].sum()
    results_df['any_violation'] = (results_df['violates_x_bounds'] | 
                                   results_df['violates_y_bounds'] | 
                                   results_df['has_intersections']).astype(int)
    
    return results_df