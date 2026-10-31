"""File contains unit tests for stage_measure."""

import logging
import numpy as np
import pytest
import tempfile
from copy import copy
from labthings_fastapi.testing import create_thing_without_server

from openflexure_microscope_server.things import stage_measure
from openflexure_microscope_server.things.camera_stage_mapping import (
    csm_img_to_stage,
    csm_stage_to_img,
)


# Useful generators
def increasing_xy_dict_generator(*_args, **_kwargs):
    """Generate x-y dictionaries of incrementing sizes.

    These don't simulate expected effects, but allow checking sequential reads of a
    function/property were used.
    """
    i = 0
    while True:
        yield {"x": i, "y": i}
        i += 1


def increasing_xyz_dict_generator(*_args, **_kwargs):
    """Generate x-y-z dictionaries of incrementing sizes.

    These don't simulate expected effects, but allow checking sequential reads of a
    function/property were used.
    """
    i = 0
    while True:
        yield {"x": i, "y": i, "z": i}
        i += 1


@pytest.fixture
def csm_matrix():
    """Return an example CSM matrix."""
    return [
        [0.03061156624485296, -1.8031242270940833],
        [1.773236372778601, 0.006660431608601435],
    ]


@pytest.fixture
def example_cam_stage_data():
    """Return some example data in a CamStageTracker."""
    mock_positions = [
        {"x": 0, "y": 0, "z": 42},
        {"x": 727, "y": 2, "z": 154},
        {"x": 1454, "y": 4, "z": 248},
        {"x": 2181, "y": 6, "z": 351},
        {"x": 2908, "y": 8, "z": 430},
        {"x": 3635, "y": 10, "z": 490},
    ]
    cam_stage_data = stage_measure.CamStageTracker()

    # loop through mock positions recording them with an offset.
    for position in mock_positions:
        offset = {"x": 54.4, "y": 0}
        cam_stage_data.record_movement(position, offset)
    return cam_stage_data


def test_predict_z(example_cam_stage_data):
    """Check that the prediction for the next z position is correct."""
    mock_z_diff = example_cam_stage_data.predict_z_displacement(
        axis="x",
        stage_movement={"x": 5243, "y": 0},
        stage_position={"x": 3635, "y": 10, "z": 490},
    )
    expected_z_diff = 153
    assert mock_z_diff == expected_z_diff


def test_find_tuning_point(example_cam_stage_data):
    """Check that the prediction for the tuning point and z position."""
    mock_turn = example_cam_stage_data.find_turning_point(axis="x")
    expected_turning = {"x": 7580, "y": 10, "z": 661}
    assert mock_turn == expected_turning


@pytest.mark.parametrize(
    ("movement", "axis", "other_axis"),
    [
        ({"x": 2908, "y": 0}, "x", "y"),
        ({"x": -29, "y": 0}, "x", "y"),
        ({"x": 0, "y": 123}, "y", "x"),
        ({"x": 0, "y": -456}, "y", "x"),
    ],
)
def test_axis_from_movement_dict(movement, axis, other_axis):
    """Test that _axis_from_movement_dict identifies the correct axes."""
    assert stage_measure._axis_from_movement_dict(movement) == axis

    ret_axes = stage_measure._axis_from_movement_dict(movement, return_other=True)
    assert ret_axes == (axis, other_axis)


def csm_img_to_stage_colprod(matrix, *, x, y):
    """Reference [stage][image] column product for test assertions."""
    import numpy as _np
    r = _np.dot(_np.array(matrix), _np.array([x, y]))
    return {"x": int(round(r[0])), "y": int(round(r[1]))}


@pytest.mark.parametrize("movement", [{"x": 0, "y": 0}, {"x": 10, "y": 10}])
def test_error_on_axis_from_movement_dict(movement):
    """Check _axis_from_movement_dict errors if both axes are zero, or both are non-zero."""
    with pytest.raises(ValueError, match="Either x or y movement should be zero"):
        assert stage_measure._axis_from_movement_dict(movement)


@pytest.mark.parametrize(
    ("par_fraction", "too_high"),
    [
        (-0.20, True),
        (-0.11, True),
        (-0.09, False),
        (-0.05, False),
        (0.00, False),
        (0.05, False),
        (0.09, False),
        (0.11, True),
        (0.20, True),
    ],
)
def test_parasitic_detect(par_fraction, too_high):
    """Check parasitic motion is detected if the fraction of parasitic motion is too high."""
    movement = {"x": 2908, "y": 0}

    offset = copy(movement)
    offset["y"] = movement["x"] * par_fraction

    detected = stage_measure._parasitic_motion_detected(
        movement=movement, offset=offset
    )
    assert detected == too_high


def test_error_if_no_stream_res_set_when_requesting_img_coords():
    """Check a RuntimeError thrown when requesting image coordinates if resolution unset."""
    stage_measure_thing = create_thing_without_server(
        stage_measure.StageMeasurementThing, mock_all_slots=True
    )
    with pytest.raises(RuntimeError, match="Stream resolution must be set"):
        stage_measure_thing._img_percentage_to_img_coords(20, "x")


@pytest.fixture
def stage_measure_thing(
        example_cam_stage_data, csm_matrix, mocker
) -> stage_measure.StageMeasurementThing:
    """Yield a StageMeasurementThing already populated with some example cam_stage_data."""
    stage_measure_thing = create_thing_without_server(
        stage_measure.StageMeasurementThing, mock_all_slots=True
    )
    type(
        stage_measure_thing._thing_server_interface
    ).application_config = mocker.PropertyMock(
        return_value={"data_folder": tempfile.gettempdir()}
    )
    stage_measure_thing._stream_resolution = [800, 600]
    stage_measure_thing._cam_stage_data = example_cam_stage_data

    def apply_csm(x: float, y: float, **_kwargs: float) -> dict[str, int]:
        """Convert image coordinates to stage coordinates."""
        return csm_img_to_stage(csm_matrix, x=x, y=y)

    def un_apply_csm(x: float, y: float, **_kwargs: float) -> dict[str, float]:
        """Convert stage coordinates to image coordinates."""
        return csm_stage_to_img(csm_matrix, x=x, y=y)

    stage_measure_thing._cam.image_is_sample.return_value = (
        True,
        "Mocked not measured.",
    )

    # Set up mock csm to return a CSM matrix
    stage_measure_thing._csm.image_to_stage_displacement_matrix = csm_matrix
    stage_measure_thing._csm.convert_image_to_stage_coordinates.side_effect = apply_csm
    stage_measure_thing._csm.convert_stage_to_image_coordinates.side_effect = (
        un_apply_csm
    )
    return stage_measure_thing


@pytest.mark.parametrize(
    ("pos", "target_pos", "axis", "expected_dir"),
    [
        ({"x": 5000, "y": 0, "z": 0}, {"x": 0, "y": 0, "z": 0}, "x", -1),
        ({"x": -5000, "y": 0, "z": 0}, {"x": 0, "y": 0, "z": 0}, "x", 1),
        ({"x": 0, "y": 0, "z": 0}, {"x": 5000, "y": 0, "z": 0}, "x", 1),
        # Y is flipped due to CSM sign
        ({"x": 0, "y": 5000, "z": 0}, {"x": 0, "y": 0, "z": 0}, "y", 1),
    ],
)
def test_img_dir_from_stage_coords(
        pos, target_pos, axis, expected_dir, stage_measure_thing
):
    """Check image direction is correctly generated."""
    stage_measure_thing._stage.position = pos
    direction = stage_measure_thing._img_dir_from_stage_coords(target_pos, axis)
    assert direction == expected_dir


def test_distance_in_img_percentage_err(stage_measure_thing):
    """Check that _distance_in_img_percentage throws error if stream res is not set."""
    stage_measure_thing._stream_resolution = None
    with pytest.raises(RuntimeError):
        stage_measure_thing._distance_in_img_percentage({"x": 0, "y": 0, "z": 0}, "x")


@pytest.mark.parametrize(
    ("pos", "target_pos", "axis", "expected_perc"),
    [
        ({"x": 5000, "y": 0, "z": 0}, {"x": 0, "y": 0, "z": 0}, "x", -352.44),
        ({"x": -5000, "y": 0, "z": 0}, {"x": 0, "y": 0, "z": 0}, "x", 352.44),
        ({"x": 0, "y": 0, "z": 0}, {"x": 5000, "y": 0, "z": 0}, "x", 352.44),
        # Y is flipped due to CSM sign
        ({"x": 0, "y": 5000, "z": 0}, {"x": 0, "y": 0, "z": 0}, "y", 462.131),
    ],
)
def test_distance_in_img_percentage(
        pos, target_pos, axis, expected_perc, stage_measure_thing
):
    """Check _distance_in_img_percentage calculates expected values based on mock CSM."""
    stage_measure_thing._stage.position = pos
    img_perc = stage_measure_thing._distance_in_img_percentage(target_pos, axis)
    assert round(img_perc, 3) == expected_perc


def test_offset_from(stage_measure_thing, mocker):
    """Check the calls and returns for StageMeasurementThing._offset_from."""
    # Set up mock for the FFT displacement
    disp_between_route = (
        "openflexure_microscope_server.things.stage_measure."
        "fft_image_tracking.displacement_between_images"
    )
    mock_disp_between = mocker.patch(
        disp_between_route,
        return_value=[123, 456],
    )

    # Run it
    offset = stage_measure_thing._offset_from(before_img="MOCK_IMAGE")

    # Check the offset is a dictionary with the correct values for the axes
    assert offset["x"] == 456
    assert offset["y"] == 123
    # Check 1 image was taken
    assert stage_measure_thing._cam.grab_as_array.call_count == 1
    mock_after_image = stage_measure_thing._cam.grab_as_array.return_value
    # Check the FFT displacement was called once
    assert mock_disp_between.call_count == 1
    displacement_kwargs = mock_disp_between.call_args.kwargs
    # Check the image inputs
    assert displacement_kwargs["image_0"] == "MOCK_IMAGE"
    assert displacement_kwargs["image_1"] == mock_after_image


@pytest.mark.parametrize("perform_autofocus", [True, False])
def test_move_and_measure(
        perform_autofocus, stage_measure_thing, mocker, csm_matrix
):
    """Test _move_and_measure with and without initial autofocus checking call counts.

    This doesn't test with the repeated autofocus if motion isn't detected
    """
    mock_offset_value = {"x": 100, "y": 3}
    mocker.patch.object(
        stage_measure_thing, "_offset_from", return_value=mock_offset_value
    )
    movement = {"x": 100, "y": 0}
    offset = stage_measure_thing._move_and_measure(
        movement=movement, perform_autofocus=perform_autofocus
    )

    # Check exactly 1 stage-space move was executed
    assert stage_measure_thing._stage.move_relative.call_count == 1
    # The movement is converted with the [stage][image] column product
    expected_stage_move = csm_img_to_stage_colprod(
        csm_matrix, x=movement["x"], y=movement["y"]
    )
    move_kwargs = stage_measure_thing._stage.move_relative.call_args.kwargs
    assert move_kwargs["x"] == expected_stage_move["x"]
    assert move_kwargs["y"] == expected_stage_move["y"]
    # Check autofocus call count
    expected_af_count = 1 if perform_autofocus else 0
    assert (
            stage_measure_thing._autofocus.looping_autofocus.call_count == expected_af_count
    )
    # And check final return
    assert offset == mock_offset_value


@pytest.mark.parametrize(
    ("x_offsets", "n_offset_measures", "expected_return"),
    [
        ([5.1, 0.2], 1, {"x": 5.1, "y": 0}),
        ([-5.1, 0.2], 1, {"x": -5.1, "y": 0}),
        ([0.1, 5.2, 0.3, 0.4, 0.5], 2, {"x": 5.2, "y": 0}),
        ([0.1, 0.2, 5.3, 0.4, 0.5], 3, {"x": 5.3, "y": 0}),
        ([0.1, 0.2, 0.3, 5.4, 0.5], 4, {"x": 5.4, "y": 0}),
        # n_offset_measures shouldn't go higher than 4 as max autofocus repeats is 3
        ([0.1, 0.2, 0.3, 0.4, 5.5], 4, {"x": 0.4, "y": 0}),
    ],
)
def test_move_and_measure_with_refocus(
        x_offsets, n_offset_measures, expected_return, stage_measure_thing, mocker, csm_matrix
):
    """Test _move_and_measure with final refocus if offset is too small."""
    return_dicts = tuple({"x": x, "y": 0} for x in x_offsets)
    offset_from_mock = mocker.patch.object(
        stage_measure_thing, "_offset_from", side_effect=return_dicts
    )
    movement = {"x": 10, "y": 0}
    offset = stage_measure_thing._move_and_measure(
        movement=movement,
        perform_autofocus=False,
        max_autofocus_repeats=3,
        abs_min_offset=5,
    )
    # Check exactly 1 stage-space move was executed
    assert stage_measure_thing._stage.move_relative.call_count == 1
    expected_stage_move = csm_img_to_stage_colprod(
        csm_matrix, x=movement["x"], y=movement["y"]
    )
    move_kwargs = stage_measure_thing._stage.move_relative.call_args.kwargs
    assert move_kwargs["x"] == expected_stage_move["x"]
    assert move_kwargs["y"] == expected_stage_move["y"]
    # Check expected _offset_from calls
    assert offset_from_mock.call_count == n_offset_measures
    # Check autofocus call count is 1 less than number of offset measures as no autofocus
    # is performed before the first one
    expected_af_count = n_offset_measures - 1
    assert (
            stage_measure_thing._autofocus.looping_autofocus.call_count == expected_af_count
    )
    # And check final return
    assert offset == expected_return


def test_move_and_measure_with_bad_refocus_args(stage_measure_thing, mocker):
    """Check error if abs_min_offset is not positive when using it to determine if to autofocus."""
    offset_from_mock = mocker.patch.object(
        stage_measure_thing, "_offset_from", return_value={"x": 0, "y": 0}
    )

    # First check with abs_min_offset not set. The default should be zero.
    with pytest.raises(ValueError, match="abs_min_offset must be positive"):
        stage_measure_thing._move_and_measure(
            movement={"x": 10, "y": 0},
            perform_autofocus=False,
            max_autofocus_repeats=3,
        )
    # Then check with abs_min_offset negative, this might happen if the the expected
    # move calculation is not made absolute.
    with pytest.raises(ValueError, match="abs_min_offset must be positive"):
        stage_measure_thing._move_and_measure(
            movement={"x": 10, "y": 0},
            perform_autofocus=False,
            max_autofocus_repeats=3,
            abs_min_offset=-123.456,
        )
    # Should have never measured and offset. Just error straight away.
    assert offset_from_mock.call_count == 0


def test_move_back_until_motion_detected(stage_measure_thing, mocker):
    """Check that _move_back_until_motion_detected is making increasing negative moves.

    The moves for this method should be in opposite direction to the direction
    specified as this is moving back after the stage reaches end of its movement.
    """
    mock_move_n_meas = mocker.patch.object(
        stage_measure_thing, "_move_and_measure", return_value={"x": 0, "y": 0}
    )

    with pytest.raises(RuntimeError, match="Cannot detect motion again"):
        stage_measure_thing._move_back_until_motion_detected("y", -1)

    max_tries = int(1.5 * stage_measure.BIG_STEP / stage_measure.SMALL_STEP)
    expected_step = 800 * stage_measure.SMALL_STEP / 100
    assert mock_move_n_meas.call_count == max_tries
    for _i, call_args in enumerate(mock_move_n_meas.call_args_list):
        call_args.kwargs["movement"] = {"x": 0, "y": expected_step}
        call_args.kwargs["perform_autofocus"] = False

    ## Reset mock and change the side effect
    mock_move_n_meas.reset_mock()
    mock_move_n_meas.side_effect = (
        {"x": 0, "y": 0},
        {"x": 0, "y": 0},
        {"x": expected_step * 0.8, "y": 0},
    )

    # Other axis and direction this time
    stage_measure_thing._move_back_until_motion_detected("x", 1)

    # Should only be called 3 times
    assert mock_move_n_meas.call_count == 3
    for _i, call_args in enumerate(mock_move_n_meas.call_args_list):
        call_args.kwargs["movement"] = {"x": -expected_step, "y": 0}
        call_args.kwargs["perform_autofocus"] = False


@pytest.mark.parametrize(
    ("good_moves", "expected_to_detect_motion", "offset_calls"),
    [
        ([0, 1, 2], True, 3),  # First 3 pass all good
        ([1, 2, 3], True, 4),  # First is bad, will refocus, still complete
        ([2, 3, 4], True, 5),  # First 2 are bad, will refocus twice, still complete
        ([3, 4, 5], True, 6),  # First 3 are bad, will refocus 3 times, still complete
        ([4, 5, 6], False, 4),  # First 4 are bad, fails
        # Check second and 3rd measurement can fail after first fails 3 times
        ([3, 5, 7], True, 8),
        ([3, 6, 9], True, 10),
        ([3, 7, 11], True, 12),
        # But they can't fail 4 times
        ([3, 8, 9], False, 8),
        ([3, 7, 12], False, 12),
    ],
)
def test_stage_still_moves(
        good_moves,
        expected_to_detect_motion,
        offset_calls,
        stage_measure_thing,
        mocker,
):
    """Test _stage_still_moves correctly detects stage movement."""
    min_offset = 800 * stage_measure.SMALL_STEP / 100 * stage_measure.DETECT_MOTION_TOL

    def gen_offsets(*_args, **_kwargs):
        """Generate offset dictionaries with small moves unless count matches ``good_moves``."""
        i = 0
        while True:
            x = min_offset * 1.2 if i in good_moves else 0.1
            yield {"x": x, "y": 0}
            i += 1

    mock_offset_from = mocker.patch.object(
        stage_measure_thing, "_offset_from", side_effect=gen_offsets()
    )

    still_moves = stage_measure_thing._stage_still_moves(axis="x", direction=1)
    assert still_moves is expected_to_detect_motion
    assert mock_offset_from.call_count == offset_calls


def test_big_z_corrected_movement(stage_measure_thing, csm_matrix):
    """Check big z corrected move moves in x/y and z the expected distances."""
    stage_measure_thing._stage.position = {"x": 5000, "y": 30, "z": 500}

    stage_measure_thing._big_z_corrected_movement("x", direction=1)

    expected_movement = {"x": 800 * stage_measure.BIG_STEP / 100, "y": 0}

    # Check there are two z-relative stage moves: the z feed-forward move
    # and the lateral stage-space move (which carries z=0 implicitly, so
    # only the feed-forward move has a z kwarg).
    expected_stage_move = csm_img_to_stage_colprod(
        csm_matrix, x=expected_movement["x"], y=0
    )
    expected_z = int(
        stage_measure_thing._cam_stage_data.predict_z_displacement(
            axis="x",
            stage_movement=expected_stage_move,
            stage_position={"x": 5000, "y": 30, "z": 500},
        )
    )
    z_moves = [
        c
        for c in stage_measure_thing._stage.move_relative.call_args_list
        if "z" in c.kwargs
    ]
    assert len(z_moves) == 1
    assert z_moves[0].kwargs["z"] == expected_z

    # The lateral move must be the stage-space conversion of the big step
    xy_moves = [
        c
        for c in stage_measure_thing._stage.move_relative.call_args_list
        if "x" in c.kwargs or "y" in c.kwargs
    ]
    assert len(xy_moves) == 1
    assert xy_moves[0].kwargs["x"] == expected_stage_move["x"]
    assert xy_moves[0].kwargs["y"] == expected_stage_move["y"]


def test_moves_for_z_prediction(stage_measure_thing, mocker, csm_matrix):
    """Check the initial moves are of the correct size and are recorded."""
    # Mock the _offset_from and stage.position to return generated dictionaries that
    # increment each time they are called. (All values 0 the first time, all values 1
    # the second time ...)
    mocker.patch.object(
        stage_measure_thing, "_offset_from", side_effect=increasing_xy_dict_generator()
    )
    # Use create=True as the Mock deosn't already have the property attr
    mocker.patch.object(
        type(stage_measure_thing._stage),
        "position",
        mocker.PropertyMock(side_effect=increasing_xyz_dict_generator()),
        create=True,
    )

    expected_movement = {"x": -800 * stage_measure.MEDIUM_STEP / 100, "y": 0}

    # Remove the mock RomData before starting
    stage_measure_thing._cam_stage_data = stage_measure.CamStageTracker()

    # Run it!
    stage_measure_thing._moves_for_z_prediction("x", direction=-1)

    # Check that _cam_stage_data now contains the 5 mocked returns in order.
    assert stage_measure_thing._cam_stage_data.offsets == [
        {"x": i, "y": i} for i in range(5)
    ]
    assert stage_measure_thing._cam_stage_data.stage_coords == [
        {"x": i, "y": i, "z": i} for i in range(5)
    ]

    # Check that the stage-space mover ran once per probe with the true
    # conversion of the expected movement
    assert stage_measure_thing._stage.move_relative.call_count == 5
    expected_stage_move = csm_img_to_stage_colprod(
        csm_matrix, x=expected_movement["x"], y=0
    )
    for arg_list in stage_measure_thing._stage.move_relative.call_args_list:
        assert arg_list.kwargs["x"] == expected_stage_move["x"]
        assert arg_list.kwargs["y"] == expected_stage_move["y"]


def test_move_until_edge_error(stage_measure_thing, mocker):
    """Check that if there is an error while moving to the edge the stage returns to start."""
    mock_position_dict = {"x": 123, "y": 456, "z": 789}
    # Use create=True as the Mock deosn't already have the property attr
    mocker.patch.object(
        type(stage_measure_thing._stage),
        "position",
        mocker.PropertyMock(return_value=mock_position_dict),
        create=True,
    )
    mocker.patch.object(
        stage_measure_thing, "_moves_for_z_prediction", side_effect=RuntimeError("Mock")
    )

    # Remove the mock RomData before starting
    stage_measure_thing._cam_stage_data = stage_measure.CamStageTracker()

    # Error should be raised even though it is in the Try:
    with pytest.raises(RuntimeError, match="Mock"):
        stage_measure_thing._move_until_edge("y", direction=-1)
    # However the "finally" should have executed, returning to the starting position
    assert stage_measure_thing._stage.move_absolute.call_count == 1
    abs_move_kwargs = stage_measure_thing._stage.move_absolute.call_args.kwargs
    expected_abs_move_kwargs = dict(
        **mock_position_dict, block_cancellation=True, backlash_compensation=None
    )
    assert abs_move_kwargs == expected_abs_move_kwargs


def test_move_until_edge(stage_measure_thing, mocker):
    """Check move until edge runs the correct movement sequence."""
    # Remove the mock RomData before starting
    stage_measure_thing._cam_stage_data = stage_measure.CamStageTracker()

    stage_measure_thing._stage.position = {"x": "mock", "y": "starting", "z": "pos"}

    def add_fake_initial_positions(*_args, **_kwargs):
        """Rather than run initial moves just add some fake data."""
        for i in range(5):
            stage_measure_thing._cam_stage_data.record_movement(
                f"mock-init-pos{i + 1}", "mock-offset"
            )

    def update_stage_pos_on_big_move(*_args, **_kwargs):
        """With big moves update the stage position."""
        big_move_count = 1
        while True:
            stage_measure_thing._stage.position = f"mocked-big-move-pos{big_move_count}"
            yield
            big_move_count += 1

    def set_final_pos(*_args, **_kwargs):
        """When move_back_until_motion_detected is run set a final stage position."""
        stage_measure_thing._stage.position = "mock-final-pos"

    # Mock the main movement functions
    mock_init_moves = mocker.patch.object(
        stage_measure_thing,
        "_moves_for_z_prediction",
        side_effect=add_fake_initial_positions,
    )
    mock_big_moves = mocker.patch.object(
        stage_measure_thing,
        "_big_z_corrected_movement",
        side_effect=update_stage_pos_on_big_move(),
    )
    mock_move_check = mocker.patch.object(
        stage_measure_thing, "_stage_still_moves", side_effect=[True] * 9 + [False]
    )
    mock_move_back = mocker.patch.object(
        stage_measure_thing,
        "_move_back_until_motion_detected",
        side_effect=set_final_pos,
    )

    # Remove the mock RomData before starting
    stage_measure_thing._cam_stage_data = stage_measure.CamStageTracker()

    # Run function
    stage_measure_thing._move_until_edge("y", direction=-1)

    # Check the call counts are as expected
    # One call of initial moves
    assert mock_init_moves.call_count == 1
    # Big moves and the check the stage is moving are each called 10 times as the mock
    # for _stage_still_moves replies False on the 10th call.
    assert mock_big_moves.call_count == 10
    assert mock_move_check.call_count == 10
    # And one call of _move_back_until_motion_detected
    assert mock_move_back.call_count == 1

    assert stage_measure_thing._cam_stage_data.stage_coords == [
        {"x": "mock", "y": "starting", "z": "pos"},
        "mock-init-pos1",
        "mock-init-pos2",
        "mock-init-pos3",
        "mock-init-pos4",
        "mock-init-pos5",
        "mocked-big-move-pos1",
        "mocked-big-move-pos2",
        "mocked-big-move-pos3",
        "mocked-big-move-pos4",
        "mocked-big-move-pos5",
        "mocked-big-move-pos6",
        "mocked-big-move-pos7",
        "mocked-big-move-pos8",
        "mocked-big-move-pos9",
        "mock-final-pos",  # This overwrites the 10th big move position recording.
    ]


def test_perform_rom_actions_locked(stage_measure_thing):
    """Check the error if running one of the calibration actions while locked."""
    # Not an RLock so no need to thread.
    stage_measure_thing._lock.acquire()
    err_msg = "Trying to run ROM test when a test is already running."
    with pytest.raises(RuntimeError, match=err_msg):
        stage_measure_thing.perform_rom_test()
    err_msg = "Trying to run recentre when a test is already running."
    with pytest.raises(RuntimeError, match=err_msg):
        stage_measure_thing.perform_recentre()


def test_perform_rom_test(stage_measure_thing, mocker):
    """Check that perform Rom Test runs the expected high level algorithm."""

    def check_and_modify_cam_stage_data(axis: str, direction: int, **_kwargs):
        """Check the cam_stage_data is empty each time, and add a final position."""
        # check cam_stage_data was cleared at the start of this run
        assert len(stage_measure_thing._cam_stage_data.stage_coords) == 0
        dist = 1111 if axis == "x" else 2222
        final_pos = {"x": 0, "y": 0}
        final_pos[axis] = dist * direction
        stage_measure_thing._cam_stage_data.stage_coords.append(final_pos)

    mock_move_until_edge = mocker.patch.object(
        stage_measure_thing,
        "_move_until_edge",
        side_effect=check_and_modify_cam_stage_data,
    )

    mocker.patch.object(
        stage_measure_thing._cam, "grab_as_array", return_value=np.zeros([123, 456, 3])
    )

    # Enter the StageMeasureThing to get the data directory.
    with stage_measure_thing as running_stage_measure_thing:
        final_dict = running_stage_measure_thing.perform_rom_test()

    # This should take less than 1 sec
    assert final_dict["Time"] <= 1
    # CSM should be 2x2 list
    assert isinstance(final_dict["CSM Matrix"], list)
    assert len(final_dict["CSM Matrix"]) == 2
    assert len(final_dict["CSM Matrix"][0]) == 2
    assert len(final_dict["CSM Matrix"][1]) == 2
    # And step range should be [2222, 4444]
    assert final_dict["Step Range"] == [2222, 4444]

    # Check that the axes were called in the expected order.
    assert mock_move_until_edge.call_count == 4

    assert mock_move_until_edge.call_args_list[0].kwargs["axis"] == "x"
    assert mock_move_until_edge.call_args_list[0].kwargs["direction"] == 1

    assert mock_move_until_edge.call_args_list[1].kwargs["axis"] == "x"
    assert mock_move_until_edge.call_args_list[1].kwargs["direction"] == -1

    assert mock_move_until_edge.call_args_list[2].kwargs["axis"] == "y"
    assert mock_move_until_edge.call_args_list[2].kwargs["direction"] == 1

    assert mock_move_until_edge.call_args_list[3].kwargs["axis"] == "y"
    assert mock_move_until_edge.call_args_list[3].kwargs["direction"] == -1


def test_perform_recenter(stage_measure_thing, mocker, caplog):
    """Check that performing the recentre runs through expected operations."""

    def check_lock(*_args, **_kwargs):
        """Check the thing is locked."""
        assert not stage_measure_thing._lock.acquire(blocking=False)

    mock_set_stream_res = mocker.patch.object(
        stage_measure_thing, "_set_stream_resolution", side_effect=check_lock
    )
    mock_recentre_axis = mocker.patch.object(stage_measure_thing, "_recentre_axis")

    # Set a mock stage position, as we patch _recentre_axis this should be logged
    # as the centre.
    stage_measure_thing._stage.position = {"x": 4321, "y": 1234, "z": 0}

    with caplog.at_level(logging.INFO):
        stage_measure_thing.perform_recentre()

    assert len(caplog.messages) == 3
    assert caplog.messages[0] == "Recentring the stage."
    assert caplog.messages[1] == "Centre is estimated at (4321, 1234)."
    assert caplog.messages[2] == "Position reset to (0, 0, 0)."

    # Check the lock was freed
    assert stage_measure_thing._lock.acquire(blocking=False)
    stage_measure_thing._lock.release()

    assert mock_set_stream_res.call_count == 1

    # Recentre called first in x then in y
    assert mock_recentre_axis.call_count == 2
    assert mock_recentre_axis.call_args_list[0].args[0] == "x"
    assert mock_recentre_axis.call_args_list[1].args[0] == "y"

    # check that the position set to zero once
    assert stage_measure_thing._stage.set_zero_position.call_count == 1


@pytest.mark.parametrize("true_on", [1, 4, 10, 11])
def test_recentre_axis(true_on, stage_measure_thing, mocker):
    """Test the high level algorithm of recentring an axis.

    This doesn't include deciding if we are centred or choosing the direction to move.
    """
    ## Start such that movement starts negative
    stage_measure_thing._stage.position = {"x": 10000, "y": 10000, "z": 0}
    mock_moves = mocker.patch.object(stage_measure_thing, "_moves_for_z_prediction")

    # Mock _recentre_decision so it returns (False, 1) a number of times then finally
    # (True, 1).
    decisions = [(False, 1)] * (true_on - 1) + [(True, 1)]
    mock_recentre_decision = mocker.patch.object(
        stage_measure_thing, "_recentre_decision", side_effect=decisions
    )

    if true_on > 10:
        with pytest.raises(RuntimeError, match="Couldn't find centre"):
            stage_measure_thing._recentre_axis("x")
    else:
        stage_measure_thing._recentre_axis("x")

    # _recentre_decision is called until True or exits after 10 attempts
    assert mock_recentre_decision.call_count == min(true_on, 10)

    # The _moves_for_z_prediction is called once before any decision, and not called
    # after a decision of centred. The max call count is 10
    assert mock_moves.call_count == min(true_on, 10)
    for i, arg_list in enumerate(mock_moves.call_args_list):
        # First direction is -ve due to starting pos, then it changes direction
        # as the mock always returns 1
        expected_dir = -1 if i < 1 else 1
        assert arg_list.kwargs["direction"] == expected_dir


@pytest.mark.parametrize(
    ("dist", "direction", "expected_decision"),
    [
        (500, 1, (False, 1)),  # Big step is 200% this is over, movement is positive.
        (-500, -1, (False, -1)),
        (200, 1, (False, 1)),
        (199, 1, (True, 1)),  # Less than 200% FOV movement return centred=True
        (-199, -1, (True, 1)),  # Always return 1 when c
    ],
)
def test_recentre_decision(
        dist, direction, expected_decision, stage_measure_thing, mocker
):
    """What the algorithm decides to do in different situations."""
    # Mock find turning point just because there is no _cam_stage_data
    mocker.patch.object(stage_measure_thing._cam_stage_data, "find_turning_point")

    # As _distance_in_img_percentage and _img_dir_from_stage_coords are tested
    # this test mocks their values and checks the return is as expected
    mocker.patch.object(
        stage_measure_thing, "_distance_in_img_percentage", return_value=dist
    )
    mocker.patch.object(
        stage_measure_thing, "_img_dir_from_stage_coords", return_value=direction
    )

    assert stage_measure_thing._recentre_decision("x") == expected_decision

    # Check that we make an absolute move (to the centre) before returning centred.
    centred = expected_decision[0]
    stage_measure_thing._stage.move_absolute.call_count == 1 if centred else 0
