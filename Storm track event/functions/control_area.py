# importng packages
import geopandas as gpd
from shapely.geometry import Point
import os
from shapely.geometry import Polygon
from shapely.affinity import scale, rotate
from shapely.ops import transform
from functools import partial
import pyproj
import numpy as np


def create_circle_shapefile(center_lon, center_lat, radius_meters, output_path):
    """
    Create a shapefile with a single circular polygon.

    Parameters:
    - center_lon: Longitude of the circle center
    - center_lat: Latitude of the circle center
    - radius_meters: Radius of the circle in meters
    - output_path: Path to the output shapefile (e.g., 'circle.shp')
    """

    # Create center point
    center = Point(center_lon, center_lat)

    # Define a projection: WGS84 -> UTM (meters)
    proj_utm = partial(
        pyproj.transform,
        pyproj.Proj(proj='latlong', datum='WGS84'),
        pyproj.Proj(proj='aeqd', datum='WGS84', lat_0=center_lat, lon_0=center_lon)
    )

    # Create buffer (in meters) around the projected center point
    circle = transform(proj_utm, center).buffer(radius_meters)

    # Project the buffered circle back to WGS84
    proj_wgs84 = partial(
        pyproj.transform,
        pyproj.Proj(proj='aeqd', datum='WGS84', lat_0=center_lat, lon_0=center_lon),
        pyproj.Proj(proj='latlong', datum='WGS84')
    )
    circle_wgs84 = transform(proj_wgs84, circle)

    # Create GeoDataFrame and save
    gdf = gpd.GeoDataFrame(geometry=[circle_wgs84], crs='EPSG:4326')
    gdf.to_file(output_path)

    print(f"Shapefile saved to: {output_path}")


def create_ellipse_shapefile(center_lon, center_lat, major_axis_m, minor_axis_m, rotation_deg, output_path):
    """
    Create a shapefile with a single elliptical polygon.

    Parameters:
    - center_lon: Longitude of the ellipse center
    - center_lat: Latitude of the ellipse center
    - major_axis_m: Length of the major axis (in meters)
    - minor_axis_m: Length of the minor axis (in meters)
    - rotation_deg: Rotation angle in degrees (counter-clockwise)
    - output_path: Path to the output shapefile (e.g., 'ellipse.shp')
    """

    # Create center point
    center = Point(center_lon, center_lat)

    # Define a local projection (Azimuthal Equidistant) centered on the ellipse
    proj_to_meters = partial(
        pyproj.transform,
        pyproj.Proj(proj='latlong', datum='WGS84'),
        pyproj.Proj(proj='aeqd', datum='WGS84', lat_0=center_lat, lon_0=center_lon)
    )

    proj_to_degrees = partial(
        pyproj.transform,
        pyproj.Proj(proj='aeqd', datum='WGS84', lat_0=center_lat, lon_0=center_lon),
        pyproj.Proj(proj='latlong', datum='WGS84')
    )

    # Project the center point to meters
    center_m = transform(proj_to_meters, center)

    # Create a unit circle and scale to ellipse
    ellipse = scale(center_m.buffer(1, resolution=100), major_axis_m, minor_axis_m)

    # Rotate the ellipse around its center
    ellipse = rotate(ellipse, rotation_deg, origin='center')

    # Project back to lat/lon
    ellipse_wgs84 = transform(proj_to_degrees, ellipse)

    # Create GeoDataFrame and save
    gdf = gpd.GeoDataFrame(geometry=[ellipse_wgs84], crs='EPSG:4326')
    gdf.to_file(output_path)

    print(f"Ellipse shapefile saved to: {output_path}")