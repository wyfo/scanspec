import base64
from dataclasses import dataclass
from typing import Any, List, Optional

import aiohttp_cors
import graphql
from aiohttp import web
from apischema.graphql import graphql_schema, resolver
from graphql_server.aiohttp.graphqlview import GraphQLView, _asyncify
from numpy import (
    absolute,
    amin,
    array2string,
    dtype,
    float64,
    frombuffer,
    linalg,
    ndarray,
)

from scanspec.core import Path
from scanspec.specs import Spec


@dataclass
class Points:
    """ A collection of singular or multidimensional locations in scan space"""

    def __init__(self, points: ndarray):
        self._points = points

    @resolver
    def string(self) -> str:
        return array2string(self._points)

    @resolver
    def float_list(self) -> List[float]:
        return self._points.tolist()

    @resolver
    def b64(self) -> str:
        # make sure the data is sent as float64
        assert dtype(self._points[0]) == dtype(float64)
        return base64.b64encode(self._points.tobytes()).decode("utf-8")

    # Self b64 decoder for testing purposes
    @resolver
    def b64Decode(self) -> str:
        r = dtype(self._points[0])
        s = base64.decodebytes(base64.b64encode(self._points.tobytes()))
        t = frombuffer(s, dtype=r)
        return array2string(t)


@dataclass
class AxisFrames:
    """ A collection of frames (comprising midpoints with lower and upper bounds)
    present in each axis of the Spec
    """

    axis: str
    """A fixed reference that can be scanned. i.e. a motor, time or
    number of repetitions.
    """
    lower: Points
    """The lower bounds of each midpoint (used when fly scanning)"""
    midpoints: Points
    """The middle points of the scan"""
    upper: Points
    """The upper bounds of each midpoint (used when fly scanning)"""
    smallest_step: float
    """The smallest step between midpoints in this axis"""


@dataclass
class SmallestStep:
    """ The smallest distance between midpoints in a multidimensional scan space"""

    def __init__(self, points: List[ndarray]):
        # points is an mxn array where n is the number of axes present in the scan
        self._points = points

    @resolver
    def absolute(self) -> float:
        # Calc abs diffs of all axes
        absolute_diffs = [abs_diffs(axis_midpoints) for axis_midpoints in self._points]
        # Return the smallest value (Aka. smallest step)
        return amin(linalg.norm(absolute_diffs, axis=0))


@dataclass
class PointsResponse:
    """ The highest level of the getPoints query, allowing users to customise their
    return data from the points present in the scan to some metadata about them
    """

    axes: List[AxisFrames]
    """A list of all of the points present in the spec per axis"""
    total_frames: int
    """The number of frames present across the entire spec"""
    returned_frames: int
    """The number of frames returned by the getPoints query
    (controlled by the max_points argument)"""
    smallest_abs_step: SmallestStep
    """The smallest step between midpoints across all axes in the scan"""


# Checks that the spec will produce a valid scan
def validate_spec(spec: Spec) -> Any:
    """ A query used to confirm whether or not the Spec will produce a viable scan"""
    # apischema will do all the validation for us
    return spec.serialize()


# Returns a full list of points for each axis in the scan
# TODO Update max_frames with a more sophisticated method of reducing scan points
def get_points(spec: Spec, max_frames: Optional[int] = 100000) -> PointsResponse:
    """A query that takes a Spec and calculates the points present in the scan
    (for each axis) plus some metadata about the points.

    Arguments:
            [spec]: [The specification of the scan]
            [max_frames]: [The maximum number of frames the user wishes to receive]

    Returns:
        [PointsResponse]: [A dataclass containing information about the scan points
                            present in the spec]
    """
    dims = spec.create_dimensions()  # Grab dimensions from spec
    path = Path(dims)  # Convert to a path

    # TOTAL FRAMES
    total_frames = len(path)  # Capture the total length of the path

    # MAX FRAMES | RETURNED FRAMES
    # Limit the consumed data by the max_frames argument
    # WARNING: path object is consumed after this statement
    if max_frames is None:
        # Return as many frames as possible
        returned_frames = len(path)
        chunk = path.consume(len(path))

    elif max_frames >= len(path):
        # Return all of the frames within that selection
        returned_frames = len(path)
        chunk = path.consume(len(path))

    else:
        # Cap the frames by the max limit
        returned_frames = max_frames
        chunk = path.consume(max_frames)

    # POINTS
    scan_points = [
        AxisFrames(
            axis,
            Points(chunk.lower[axis]),
            Points(chunk.midpoints[axis]),
            Points(chunk.upper[axis]),
            float(amin(abs_diffs(chunk.midpoints[axis]))),
        )
        for axis in spec.axes()
    ]

    # ABSOLUTE SMALLEST STEP
    smallest_abs_step = SmallestStep(list(chunk.midpoints.values()))

    return PointsResponse(
        scan_points, total_frames, returned_frames, smallest_abs_step,
    )


def abs_diffs(array: ndarray) -> ndarray:
    """Calculates the absolute differences between adjacent elements in the array
       used as part of the smallest step calculation for each axis

    Args:
        array (ndarray): A 1xN array of numerical values

    Returns:
        ndarray: A newly constucted array of absolute differences
    """
    # [array[1] - array[0], array[2] - array[1], ...]
    adjacent_diffs = array[1:] - array[:-1]
    return absolute(adjacent_diffs)


# Define the schema
schema = graphql_schema(query=[validate_spec, get_points])


def schema_text() -> str:
    return graphql.utilities.print_schema(schema)


def run_app(cors=False, port=8080):
    app = web.Application()

    view = GraphQLView(schema=schema, graphiql=True)

    # Make GraphQLView compatible with aiohttp-cors
    # https://github.com/aio-libs/aiohttp-cors/issues/241#issuecomment-514278001
    for method in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"):  # no OPTIONS
        app.router.add_route(method, "/graphql", _asyncify(view), name="graphql")

    # Optional, for adding batch query support (used in Apollo-Client)
    # GraphQLView.attach(app, schema=schema, batch=True, route_path="/graphql/batch")

    if cors:
        # Configure default CORS settings.
        cors_config = aiohttp_cors.setup(
            app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True, expose_headers="*", allow_headers="*",
                )
            },
        )

        # Configure CORS on all routes.
        for route in list(app.router.routes()):
            cors_config.add(route)

    web.run_app(app, port=port)
